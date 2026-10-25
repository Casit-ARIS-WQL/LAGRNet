"""Section 3.6: five active objectives; missing branches are errors, not zeros."""
import torch
import torch.nn as nn
from geometry import inverse_homography, warp


def masked_l1(pred, target, mask):
    mask = mask.bool().expand_as(pred)
    values = []
    for p, t, m in zip(pred, target, mask):
        if not m.any():
            raise ValueError('A sample has no valid pixels for this loss')
        values.append((p[m] - t[m]).abs().mean())
    return torch.stack(values).mean()


def affine_invariant_depth_loss(pred, target, mask, eps=1e-6):
    """Per-image median/MAD normalization on the same valid annotations.

    eps extends the manuscript's nondegenerate definition numerically; it does
    not discard constant predictions or erase their depth-supervision gradient.
    """
    values = []
    for p, t, m in zip(pred, target, mask.bool()):
        if not m.any():
            raise ValueError('Depth supervision requires annotated pixels in every sample')
        p, t = p[m], t[m]
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite valid depth supervision')
        pc, tc = p - p.median(), t - t.median()
        pn = pc / pc.abs().mean().clamp_min(eps)
        tn = tc / tc.abs().mean().clamp_min(eps)
        values.append((pn - tn).abs().mean())
    return torch.stack(values).mean()


class LAGRLoss(nn.Module):
    def __init__(self, lam_pho=1.0, lam_grp=0.5, lam_sheaf=0.1, lam_sm=0.01, gamma=1.0):
        super().__init__()
        self.weights = {'depth': 1., 'pho': lam_pho, 'grp': lam_grp,
                        'sheaf': lam_sheaf, 'sm': lam_sm}
        self.gamma = gamma
        if any(v < 0 for v in self.weights.values()):
            raise ValueError('Loss weights must be nonnegative')

    @staticmethod
    def photometric_loss(image, reconstructions, masks):
        if not reconstructions or len(reconstructions) != len(masks):
            raise ValueError('Each synthetic reconstruction requires a validity mask')
        if torch.is_grad_enabled() and not any(x.requires_grad for x in reconstructions):
            raise ValueError('Photometric reconstruction is detached from trainable geometry')
        return torch.stack([masked_l1(x, image, mask)
                            for x, mask in zip(reconstructions, masks)]).mean()

    @staticmethod
    def group_loss(pred, transformed_pred, transform, transformed_valid):
        aligned, valid = warp(transformed_pred, inverse_homography(transform),
                              source_valid=transformed_valid)
        return masked_l1(aligned, pred, valid)

    def smoothness_loss(self, pred, image):
        dx = (pred[..., 1:] - pred[..., :-1]).abs()
        dy = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
        ix = (image[..., 1:] - image[..., :-1]).abs().mean(1, keepdim=True)
        iy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, keepdim=True)
        # The manuscript sums spatial derivatives. Only batch is averaged.
        return ((dx * (-self.gamma * ix).exp()).sum() +
                (dy * (-self.gamma * iy).exp()).sum()) / pred.shape[0]

    def forward(self, outputs, batch):
        pred = outputs['inverse_depth']
        parts = {'depth': affine_invariant_depth_loss(pred, batch['inverse_depth'], batch['valid_mask'])}
        if self.weights['pho']:
            parts['pho'] = self.photometric_loss(batch['image_raw'], batch['reconstructions'],
                                                batch['reconstruction_masks'])
        if self.weights['grp']:
            parts['grp'] = self.group_loss(pred, batch['transformed_prediction'],
                                           batch['group_transform'], batch['transformed_valid'])
        if self.weights['sheaf']:
            parts['sheaf'] = outputs['sheaf_energy'].mean() / outputs['N_patches']
        if self.weights['sm']:
            parts['sm'] = self.smoothness_loss(pred, batch['image_raw'])
        # Zero weights explicitly specify an ablation, never a silent missing branch.
        total = sum(self.weights[name] * value for name, value in parts.items())
        return total, dict(parts, total=total)
