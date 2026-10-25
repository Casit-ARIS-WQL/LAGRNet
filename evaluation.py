"""Relative inverse-depth metrics, with explicit scoring and failure policy."""
import torch


def align_scale_shift(prediction, target, valid):
    p, t = prediction[valid].double(), target[valid].double()
    if p.numel() < 2 or not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('Alignment requires at least two finite evaluation pixels')
    pc = p - p.mean()
    variance = pc.square().sum()
    if variance <= torch.finfo(p.dtype).eps * max(1., p.square().sum().item()):
        raise ValueError('Constant/degenerate prediction cannot determine scale and shift')
    a = (pc * (t - t.mean())).sum() / variance
    b = t.mean() - a * p.mean()
    return prediction.double() * a + b, a, b


def inverse_depth_metrics(prediction, metric_depth, valid, nonpositive='error'):
    """Fit least-squares a,b on all GT-valid pixels, then score inverse depth.

    Default rejects nonpositive aligned predictions rather than silently improving
    ratios by dropping errors. 'exclude' implements the manuscript's positive-pair
    mask explicitly and returns coverage; 'clamp' is a separate declared convention.
    The selected policy and positive-pixel coverage are recorded in evaluation output.
    """
    valid = valid.bool() & torch.isfinite(metric_depth) & (metric_depth > 0)
    target = torch.zeros_like(metric_depth, dtype=torch.float64)
    target[valid] = metric_depth[valid].double().reciprocal()
    aligned, a, b = align_scale_shift(prediction, target, valid)
    p, t = aligned[valid], target[valid]
    positive = p > 0
    coverage = positive.double().mean().item()
    if nonpositive == 'error' and not positive.all():
        raise ValueError('Nonpositive aligned inverse depth; choose and report an explicit evaluation policy')
    elif nonpositive == 'exclude':
        p, t = p[positive], t[positive]
    elif nonpositive == 'clamp':
        p = p.clamp_min(1e-6)
    elif nonpositive != 'error':
        raise ValueError('Unknown nonpositive policy')
    if not p.numel():
        raise ValueError('No positive aligned prediction/GT pairs')
    delta = torch.maximum(p / t, t / p)
    diff = p - t
    return {'abs_rel': (diff.abs()/t).mean().item(), 'sq_rel': (diff.square()/t).mean().item(),
            'rmse': diff.square().mean().sqrt().item(),
            'rmse_log': (p.log()-t.log()).square().mean().sqrt().item(),
            'a1': (delta < 1.25).double().mean().item(),
            'a2': (delta < 1.25**2).double().mean().item(),
            'a3': (delta < 1.25**3).double().mean().item(),
            'positive_fraction': coverage, 'n_valid': int(valid.sum()),
            'n_scored': p.numel(), 'scale': a.item(), 'shift': b.item()}


METRICS = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')


def summarize_records(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row['dataset'], []).append(row)
    datasets = {name: {k: sum(r[k] for r in rows)/len(rows) for k in METRICS}
                for name, rows in grouped.items()}
    result = {'datasets': datasets, 'counts': {k: len(v) for k, v in grouped.items()},
              'aggregation': 'per-image mean, then equally weighted dataset means'}
    if set(datasets) == {'KITTI', 'NYUv2', 'ETH3D'}:
        result['three_dataset_mean'] = {k: sum(v[k] for v in datasets.values())/3 for k in METRICS}
    return result
