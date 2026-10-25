"""Score the explicitly supplied held-out splits, never use them to train."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets.relative_depth import RelativeDepthDataset
from evaluation import inverse_depth_metrics, summarize_records
from inference import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True, action='append')
    parser.add_argument('--output', default='outputs/evaluation.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--nonpositive', choices=('error', 'exclude', 'clamp'), default='error')
    parser.add_argument('--diagnostics', action='store_true', help='Record actual learned and ordinary graph energies')
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    dataset = RelativeDepthDataset(args.manifest, training=False, short_side=config.short_side)
    records = []
    for batch in dataset:
        outputs = model(batch['image'][None].to(device), diagnostics=args.diagnostics)
        q = F.interpolate(outputs['inverse_depth'], size=batch['depth'].shape[-2:],
                          mode='bilinear', align_corners=False)[0].cpu()
        metrics = inverse_depth_metrics(q, batch['depth'], batch['valid_mask'], args.nonpositive)
        record = dict(dataset=batch['dataset'], path=batch['path'], split=batch['split'], **metrics)
        if args.diagnostics:
            n = outputs['N_patches']
            record['learned_sheaf_energy_per_node'] = (outputs['sheaf_energy']/n).item()
            record['graph_energy_per_node'] = (outputs['graph_energy']/n).item()
        records.append(record)
    result = summarize_records(records)
    result.update(records=records, nonpositive_policy=args.nonpositive,
                  alignment='per-image least squares in inverse depth on all GT-valid masked pixels',
                  log_metric='natural logarithm',
                  validity='finite positive GT, intersect optional explicit mask and depth bounds from manifest',
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  manifests={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                             for p in args.manifest})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
