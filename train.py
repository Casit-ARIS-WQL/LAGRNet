"""Labeled-only, iteration-based zero-shot training. No benchmark test tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from configs.paper import PaperConfig, CHECKPOINT_FORMAT_VERSION
from datasets.relative_depth import RelativeDepthDataset
from engine import objective, polynomial_lr
from losses import LAGRLoss
from model import LAGRNet


def distributed_model(model, device):
    # Two forwards feed one objective. Re-broadcasting BN/Cayley buffers on the
    # second forward mutates tensors saved by the first autograd graph.
    # Keep per-rank BatchNorm statistics; gradients still synchronize normally.
    return DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                   broadcast_buffers=False)


def load_config(path=None):
    values = json.loads(Path(path).read_text()) if path else {}
    return PaperConfig(**values)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def provenance(manifests):
    result = {'manifests': {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in manifests}, 'torch': str(torch.__version__)}
    try:
        result['base_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result['base_commit'] = None
    # Record source contents, including uncommitted changes, not just upstream HEAD.
    root = Path(__file__).parent
    source_files = list(root.glob('*.py')) + list((root/'configs').glob('*.py')) + list((root/'datasets').glob('*.py'))
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source_files)}
    paper = root.parent / 'jssc-module.tex'
    if paper.is_file():
        result['manuscript_sha256'] = hashlib.sha256(paper.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', required=True, help='JSONL, repeat for separate sources')
    parser.add_argument('--config', help='JSON overrides of configs/paper.py')
    parser.add_argument('--output', default='runs/paper')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Stop early without changing the LR schedule horizon')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    if cfg.batch_size <= 0 or cfg.batch_size % world:
        raise ValueError('Global batch size must be positive and divisible by world size')
    if cfg.total_steps <= 0 or cfg.save_every <= 0 or cfg.log_every <= 0:
        raise ValueError('Step and logging intervals must be positive')
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    dataset = RelativeDepthDataset(args.manifest, training=True, short_side=cfg.short_side, jitter=cfg.jitter)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, seed=cfg.seed, drop_last=True)
    generator = torch.Generator().manual_seed(cfg.seed + rank)
    loader = DataLoader(dataset, batch_size=cfg.batch_size // world, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True, worker_init_fn=seed_worker,
                        generator=generator, pin_memory=device.type == 'cuda')
    if len(loader) == 0:
        raise ValueError('Fewer training samples than the global batch size')
    kwargs = cfg.model_kwargs()
    if args.resume:
        kwargs['pretrained_backbone'] = False  # all weights restored below
    model = LAGRNet(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = LAGRLoss(cfg.lam_pho, cfg.lam_grp, cfg.lam_sheaf, cfg.lam_sm, cfg.gamma_sm)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION:
            raise ValueError('Checkpoint predates learned RCL Fuse and the patch-intersection sheaf graph (format 3)')
        if checkpoint['config'] != cfg.to_dict():
            raise ValueError('Resume config differs from the saved experiment configuration')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
    run_dir = Path(args.output)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume and (run_dir / 'latest.pt').exists():
            raise FileExistsError('Output already contains checkpoints; use --resume or a new directory')
    metadata = provenance(args.manifest)
    if args.resume and checkpoint['provenance']['manifests'] != metadata['manifests']:
        raise ValueError('Training manifests changed since the checkpoint')
    if rank == 0:
        (run_dir / 'config.json').write_text(json.dumps(cfg.to_dict(), indent=2))
        (run_dir / 'provenance.json').write_text(json.dumps(metadata, indent=2))
    core = model
    if world > 1:
        model = distributed_model(model, device)
    model.train()
    stop = min(cfg.total_steps, args.stop_after if args.stop_after is not None else cfg.total_steps)
    if stop <= step:
        raise ValueError('Stop step must be greater than resumed step')
    epoch, offset = divmod(step, len(loader))
    # Recreate deterministic sample order. Worker augmentation streams are reseeded
    # at an epoch boundary; resume is not claimed to be bitwise identical.
    while step < stop:
        sampler.set_epoch(epoch)
        generator.manual_seed(cfg.seed + rank + epoch * world)
        for batch_index, batch in enumerate(loader):
            if batch_index < offset:
                continue
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            lr = polynomial_lr(step, cfg.total_steps, cfg.learning_rate, cfg.poly_power)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            total, parts, _ = objective(model, criterion, batch, cfg.augmentation_magnitude)
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite loss; no optimizer step applied')
            total.backward()
            for parameter in core.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient; no optimizer step applied')
            optimizer.step()
            step += 1
            if step == 1 or step % cfg.log_every == 0 or step == stop:
                names = list(parts)
                values = torch.stack([parts[k].detach() for k in names])
                if world > 1:
                    dist.all_reduce(values)
                    values /= world
                if rank == 0:
                    record = dict(step=step, lr=lr, **dict(zip(names, values.cpu().tolist())))
                    with (run_dir / 'losses.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
            if rank == 0 and (step % cfg.save_every == 0 or step == stop):
                state = {'format_version': CHECKPOINT_FORMAT_VERSION, 'step': step, 'config': cfg.to_dict(),
                         'model': core.state_dict(), 'optimizer': optimizer.state_dict(),
                         'provenance': metadata}
                tmp = run_dir / 'latest.tmp'
                torch.save(state, tmp)
                tmp.replace(run_dir / 'latest.pt')
                if step % cfg.save_every == 0 or step == cfg.total_steps:
                    torch.save(state, run_dir / f'step_{step:06d}.pt')
            if step >= stop:
                break
        epoch, offset = epoch + 1, 0
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
