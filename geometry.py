"""Pixel-coordinate projective warps. H maps source pixels to destination pixels.

Only spatial resampling uses align_corners=True (with its matching normalization).
Feature/image resizing elsewhere uses align_corners=False. No pose or metric depth
is inferred by these image-plane homographies.
"""
import torch
import torch.nn.functional as F


def inverse_homography(matrix):
    if matrix.shape[-2:] != (3, 3) or not torch.isfinite(matrix).all():
        raise ValueError('Homographies must be finite 3 x 3 matrices')
    inv, info = torch.linalg.inv_ex(matrix)
    if (info != 0).any() or not torch.isfinite(inv).all():
        raise ValueError('Singular homography; no valid PGL(3) action exists')
    return inv


def homography_grid(matrix, source_hw, target_hw=None, eps=1e-6):
    """Return inverse-sampling grid and full-support validity mask, Bx1xHxW."""
    target_hw = target_hw or source_hw
    h, w = target_hw
    sh, sw = source_hw
    ys, xs = torch.meshgrid(torch.arange(h, device=matrix.device, dtype=matrix.dtype),
                            torch.arange(w, device=matrix.device, dtype=matrix.dtype), indexing='ij')
    base = torch.stack((xs, ys, torch.ones_like(xs))).flatten(1)
    # Normalize the representative before denominator thresholding. Otherwise
    # very large nonzero scalar multiples would incorrectly invalidate a warp.
    scale = matrix.abs().amax(dim=(-2, -1), keepdim=True)
    if (scale == 0).any() or not torch.isfinite(scale).all():
        raise ValueError('Singular or nonfinite homography representative')
    src = inverse_homography(matrix / scale) @ base
    z = src[:, 2]
    # Preserve sign: H and -H represent the same projective transformation.
    safe_z = torch.where(z.abs() >= eps, z, torch.where(z < 0, -eps, eps))
    x, y = src[:, 0] / safe_z, src[:, 1] / safe_z
    valid = (z.abs() >= eps) & torch.isfinite(x) & torch.isfinite(y)
    tol = 1e-4
    valid &= (x >= -tol) & (x <= sw - 1 + tol) & (y >= -tol) & (y <= sh - 1 + tol)
    xn = 2 * x / (sw - 1) - 1 if sw > 1 else torch.zeros_like(x)
    yn = 2 * y / (sh - 1) - 1 if sh > 1 else torch.zeros_like(y)
    grid = torch.stack((xn, yn), -1).reshape(-1, h, w, 2)
    valid = valid.reshape(-1, 1, h, w)
    grid = torch.where(valid.permute(0, 2, 3, 1), grid, grid.new_full((), 2.0))
    return grid, valid


def warp(image, matrix, target_hw=None, source_valid=None):
    grid, valid = homography_grid(matrix, image.shape[-2:], target_hw)
    result = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if source_valid is not None:
        support = F.grid_sample(source_valid.to(image.dtype), grid,
                                mode='bilinear', padding_mode='zeros', align_corners=True)
        valid = valid & (support >= 1 - 1e-5)
    result = torch.where(valid, result, torch.zeros_like(result))
    return result, valid


def resize_homography(matrix, old_hw, new_hw):
    """Conjugate by the pixel-center resize map (align_corners=False)."""
    sy, sx = new_hw[0] / old_hw[0], new_hw[1] / old_hw[1]
    scale = matrix.new_tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return scale @ matrix @ torch.linalg.inv(scale)


def random_homography(batch, hw, device, dtype, magnitude=0.05):
    """Near-identity augmentation in normalized image coordinates.

    The magnitude controls uniform perturbations in normalized image coordinates.
    """
    if not 0 <= magnitude < 0.25:
        raise ValueError('augmentation magnitude must be in [0, 0.25)')
    matrix = torch.eye(3, device=device, dtype=dtype).repeat(batch, 1, 1)
    matrix += (2 * torch.rand_like(matrix) - 1) * magnitude
    matrix[:, 2, 2] = 1
    h, w = hw
    pixel = matrix.new_tensor([[max(w-1, 1)/2, 0, (w-1)/2],
                               [0, max(h-1, 1)/2, (h-1)/2], [0, 0, 1]])
    return pixel @ matrix @ torch.linalg.inv(pixel)


def synthetic_view_reconstructions(image, homographies):
    """I_s = rho(theta_s) I; reconstruct with rho(theta_s^{-1}).

    Keep both paths attached to learned GFM matrices. In the ideal continuous
    setting the round trip is identity; this term regularizes discrete resampling
    consistency, and is not independent metric-depth/pose supervision.
    """
    if homographies.ndim != 4 or homographies.shape[1] == 0:
        raise ValueError('Expected B x K x 3 x 3 learned view transforms, K > 0')
    reconstructions, masks = [], []
    for matrix in homographies.unbind(1):
        source, source_valid = warp(image, matrix)
        recon, valid = warp(source, inverse_homography(matrix), source_valid=source_valid)
        reconstructions.append(recon)
        masks.append(valid)
    return reconstructions, masks
