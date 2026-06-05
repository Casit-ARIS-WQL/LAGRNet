"""
Model complexity analysis script for LAGRNet.

Computes:
  - Total / trainable parameter counts
  - FLOPs and MACs (via thop or torchinfo)
  - Peak GPU/CPU memory estimate for inference
  - Per-module parameter breakdown
  - Model size on disk (float32 / float16)

Usage:
    # Default resolution (352 x 1216, KITTI), no pretrained backbone
    python model_complexity.py

    # Custom resolution
    python model_complexity.py --height 480 --width 640

    # Load a saved checkpoint
    python model_complexity.py --checkpoint checkpoints/best.pth

    # Use a GPU
    python model_complexity.py --gpu 0

    # Control model hyper-parameters
    python model_complexity.py --unified_channels 256 --K_orbit 8 \
                               --D_grade 1 --sheaf_dim 128
"""

import argparse
import sys
import os
import math
import logging

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from model import LAGRNet
from configs.kitti_config import KITTIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human(n: float, suffix: str = '') -> str:
    """Format a large number with K / M / G suffixes."""
    for unit in ('', 'K', 'M', 'G', 'T'):
        if abs(n) < 1000.0:
            return f'{n:7.2f} {unit}{suffix}'
        n /= 1000.0
    return f'{n:.2f} P{suffix}'


def count_parameters(model: nn.Module):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module, dtype=torch.float32) -> float:
    """Estimate model weight memory in MB for the given dtype."""
    bytes_per_elem = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.float64: 8}
    bpe = bytes_per_elem.get(dtype, 4)
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    return (total_params + total_buffers) * bpe / (1024 ** 2)


def per_module_params(model: nn.Module, top_k: int = 20):
    """
    Return list of (name, param_count) sorted by descending param_count
    for all named sub-modules that own *direct* parameters.
    """
    rows = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own > 0:
            rows.append((name or '(root)', own))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


# ---------------------------------------------------------------------------
# FLOPs / MACs via thop
# ---------------------------------------------------------------------------

def flops_with_thop(model: nn.Module, dummy_input: torch.Tensor):
    """
    Compute FLOPs and parameter count using the `thop` library.
    Returns (macs, params) or None if thop is not installed.
    """
    try:
        from thop import profile, clever_format
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs, params
    except ImportError:
        return None
    except Exception as exc:
        logging.warning(f'thop profile failed: {exc}')
        return None


# ---------------------------------------------------------------------------
# Summary via torchinfo
# ---------------------------------------------------------------------------

def summary_with_torchinfo(model: nn.Module, input_size: tuple, device: torch.device):
    """
    Print a detailed layer-by-layer summary using `torchinfo`.
    Returns the ModelStatistics object or None if torchinfo is unavailable.
    """
    try:
        from torchinfo import summary
        stats = summary(
            model,
            input_size=input_size,
            device=device,
            verbose=0,                   # suppress default print
            col_names=('input_size', 'output_size', 'num_params', 'mult_adds'),
            row_settings=('var_names',),
        )
        return stats
    except ImportError:
        return None
    except Exception as exc:
        logging.warning(f'torchinfo summary failed: {exc}')
        return None





def measure_latency(model: nn.Module, dummy_input: torch.Tensor,
                    device: torch.device, warmup: int = 10, runs: int = 50) -> dict:
    """
    Measure inference latency (ms) over `runs` forward passes after `warmup` passes.
    Returns min / mean / max / std latency in milliseconds, and FPS.
    """
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        times = []
        for _ in range(runs):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - t0) * 1000.0)

    times_t = torch.tensor(times)
    return {
        'min_ms':  times_t.min().item(),
        'mean_ms': times_t.mean().item(),
        'max_ms':  times_t.max().item(),
        'std_ms':  times_t.std().item(),
        'fps':     1000.0 / times_t.mean().item(),
    }

# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory_mb(model: nn.Module, dummy_input: torch.Tensor,
                       device: torch.device) -> dict:
    """
    Estimate peak memory during a single forward pass.

    Weights memory  : sum of all parameters + buffers (float32).
    Activation memory: measured via torch.cuda.max_memory_allocated delta
                       (GPU only); approximated on CPU.
    Gradient memory : roughly equal to weights memory (for training).
    """
    param_mb = model_size_mb(model, torch.float32)

    activation_mb = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize(device)
        after = torch.cuda.max_memory_allocated(device)
        activation_mb = (after - before) / (1024 ** 2)

    return {
        'weights_mb': param_mb,
        'activation_mb': activation_mb,
        'gradient_mb': param_mb,          # rough estimate (same size as weights)
        'total_training_mb': param_mb * 3  # weights + grads + optimizer (Adam ≈ 2×)
                             + (activation_mb or 0),
    }


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse(args):
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    log = logging.getLogger(__name__)

    # ---- Device ----
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    log.info(f'\nDevice : {device}\n{"=" * 60}')

    # ---- Build model ----
    model = LAGRNet(
        unified_channels=args.unified_channels,
        K_orbit=args.K_orbit,
        D_grade=args.D_grade,
        sheaf_dim=args.sheaf_dim,
        patch_grid=(args.patch_grid_h, args.patch_grid_w),
        pretrained_backbone=False,         # avoid internet download for analysis
        img_size=(args.height, args.width),
    ).to(device)
    model.eval()

    # ---- Optionally load checkpoint ----
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        log.info(f'Loaded checkpoint : {args.checkpoint}')

    # ---- Dummy input ----
    H, W = args.height, args.width
    dummy = torch.randn(1, 3, H, W, device=device)
    input_size = (1, 3, H, W)

    # ---- Parameter counts ----
    total_p, train_p = count_parameters(model)
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    log.info('\n[1] Parameter Counts')
    log.info(f'  Total parameters       : {_human(total_p)} ({total_p:,})')
    log.info(f'  Trainable parameters   : {_human(train_p)} ({train_p:,})')
    log.info(f'  Non-trainable params   : {_human(total_p - train_p)} ({total_p - train_p:,})')
    log.info(f'  Backbone parameters    : {_human(backbone_p)} ({backbone_p:,})  '
             f'({backbone_p / max(total_p, 1) * 100:.1f}% of total)')

    # ---- Model size on disk ----
    log.info('\n[2] Model Size on Disk')
    log.info(f'  Float32 (fp32)         : {model_size_mb(model, torch.float32):.2f} MB')
    log.info(f'  Float16 (fp16)         : {model_size_mb(model, torch.float16):.2f} MB')

    # ---- FLOPs via thop ----
    log.info('\n[3] FLOPs / MACs  (via thop)')
    thop_result = flops_with_thop(model, dummy)
    if thop_result is not None:
        macs, _ = thop_result
        flops = macs * 2          # 1 MAC = 2 FLOPs (multiply + add)
        log.info(f'  MACs                   : {_human(macs, "MACs")}')
        log.info(f'  FLOPs (≈ 2 × MACs)    : {_human(flops, "FLOPs")}')
        log.info(f'  GFLOPs                 : {flops / 1e9:.2f}')
    else:
        log.info('  thop not installed – run:  pip install thop')

    # ---- torchinfo summary ----
    log.info('\n[4] Layer-by-layer Summary  (via torchinfo)')
    ti_stats = summary_with_torchinfo(model, input_size, device)
    if ti_stats is not None:
        log.info(str(ti_stats))
        log.info(f'  Total MACs  (torchinfo): {_human(ti_stats.total_mult_adds, "MACs")}')
        log.info(f'  GFLOPs      (torchinfo): {ti_stats.total_mult_adds * 2 / 1e9:.2f}')
    else:
        log.info('  torchinfo not installed – run:  pip install torchinfo')

    # ---- Memory estimates ----
    log.info('\n[5] Memory Estimates')
    mem = estimate_memory_mb(model, dummy, device)
    log.info(f'  Weight memory (fp32)   : {mem["weights_mb"]:.2f} MB')
    if mem['activation_mb'] is not None:
        log.info(f'  Activation memory      : {mem["activation_mb"]:.2f} MB  '
                 f'(measured on GPU, batch=1)')
        log.info(f'  Est. inference memory  : {mem["weights_mb"] + mem["activation_mb"]:.2f} MB')
    else:
        log.info('  Activation memory      : N/A (GPU not used; re-run with --gpu 0 on CUDA)')
    log.info(f'  Est. training memory   : {mem["total_training_mb"]:.2f} MB  '
             f'(fp32 weights + grads + Adam states + activations)')

    # ---- Per-module parameter breakdown ----
    log.info('\n[6] Top-20 Modules by Parameter Count')
    log.info(f'  {"Module":<45}  {"Params":>12}  {"Share":>7}')
    log.info('  ' + '-' * 68)
    for name, cnt in per_module_params(model, top_k=20):
        share = cnt / max(total_p, 1) * 100
        log.info(f'  {name:<45}  {cnt:>12,}  {share:>6.2f}%')

    # ---- Input / Output shapes ----
    log.info('\n[7] Input / Output Shapes')
    log.info(f'  Input  : (1, 3, {H}, {W})')
    with torch.no_grad():
        out = model(dummy)
    log.info(f'  depth  : {tuple(out["depth"].shape)}')
    log.info(f'  sheaf_energy : {tuple(out["sheaf_energy"].shape)}')
    log.info(f'  F_gfm  : {tuple(out["F_gfm"].shape)}')

    # ---- Inference latency ----
    log.info('\n[8] Inference Latency  (batch=1, warmup=10, runs=50)')
    lat = measure_latency(model, dummy, device)
    log.info(f'  Min    : {lat["min_ms"]:8.2f} ms')
    log.info(f'  Mean   : {lat["mean_ms"]:8.2f} ms')
    log.info(f'  Max    : {lat["max_ms"]:8.2f} ms')
    log.info(f'  Std    : {lat["std_ms"]:8.2f} ms')
    log.info(f'  FPS    : {lat["fps"]:8.2f}')

    log.info('\n' + '=' * 60)
    log.info('Analysis complete.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    cfg = KITTIConfig()
    parser = argparse.ArgumentParser(
        description='LAGRNet model complexity analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input resolution
    parser.add_argument('--height', type=int, default=cfg.height,
                        help='Input image height')
    parser.add_argument('--width',  type=int, default=cfg.width,
                        help='Input image width')

    # Model hyper-parameters
    parser.add_argument('--unified_channels', type=int, default=cfg.unified_channels)
    parser.add_argument('--K_orbit',          type=int, default=cfg.K_orbit)
    parser.add_argument('--D_grade',          type=int, default=cfg.D_grade)
    parser.add_argument('--sheaf_dim',        type=int, default=cfg.sheaf_dim)
    parser.add_argument('--patch_grid_h',     type=int, default=cfg.patch_grid[0])
    parser.add_argument('--patch_grid_w',     type=int, default=cfg.patch_grid[1])

    # Checkpoint & device
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to .pth checkpoint (optional)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU id (-1 for CPU)')

    return parser.parse_args()


if __name__ == '__main__':
    analyse(parse_args())
