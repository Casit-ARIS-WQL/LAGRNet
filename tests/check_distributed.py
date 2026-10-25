"""Optional two-process Gloo integration check, using file rendezvous.

Run: GLOO_SOCKET_IFNAME=lo0 python tests/check_distributed.py --output report.json
On Linux omit GLOO_SOCKET_IFNAME or select the relevant local interface.
This requires permission for local inter-process communication. No benchmark data.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_pipeline import small_model
from datasets.relative_depth import normalize_rgb
from engine import objective
from losses import LAGRLoss
from train import distributed_model


def worker(rank, store, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=2)
    torch.manual_seed(100 + rank)
    core = small_model()
    model = distributed_model(core, torch.device('cpu'))
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-4)
    # Two steps also catch reducer state left incomplete after a first backward.
    for _ in range(2):
        image = torch.rand(2, 3, 96, 128)
        batch = {'image': normalize_rgb(image), 'image_raw': image,
                 'inverse_depth': torch.rand(2, 1, 96, 128),
                 'valid_mask': torch.ones(2, 1, 96, 128, dtype=torch.bool)}
        optimizer.zero_grad(set_to_none=True)
        total, parts, _ = objective(model, LAGRLoss(), batch)
        total.backward()
        assert all(parts[k] > 0 for k in ('depth','pho','grp','sheaf','sm'))
        assert core.rcl_fuse0.fuse_logits.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
        gradient = next(core.parameters()).grad.clone()
        copies = [torch.zeros_like(gradient) for _ in range(2)]
        dist.all_gather(copies, gradient)
        torch.testing.assert_close(copies[0], copies[1])
        optimizer.step()
    if rank == 0:
        record = {'world_size': 2, 'backend': 'gloo', 'init': 'file store',
                  'fixture': 'explicit small backbone, random images; not a benchmark',
                  'five_losses': sorted(parts), 'broadcast_buffers': False,
                  'gradient_synchronized': True, 'optimizer_steps': 2,
                  'learned_fuse_gradient': True, 'sheaf_edges': core.sm.edge_index.shape[1]}
        Path(output).write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='lagrnet-ddp-') as directory:
        mp.spawn(worker, args=(str(Path(directory)/'store'), str(Path(args.output).resolve())),
                 nprocs=2, join=True)
