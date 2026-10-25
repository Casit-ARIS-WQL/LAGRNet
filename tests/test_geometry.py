import pytest
import torch
from geometry import warp, inverse_homography, synthetic_view_reconstructions, resize_homography
from losses import LAGRLoss


def test_projective_scale_sign_and_identity():
    image = torch.rand(2, 3, 11, 17, dtype=torch.float64)
    identity = torch.eye(3, dtype=image.dtype).repeat(2, 1, 1)
    a, mask = warp(image, identity)
    b, mask_b = warp(image, -7 * identity)
    torch.testing.assert_close(a, image)
    torch.testing.assert_close(a, b)
    assert mask.all() and mask_b.all()
    large, large_mask = warp(image, identity * 1e12)
    torch.testing.assert_close(large, image)
    assert large_mask.all()


def test_signed_projective_denominator_and_invalid_region():
    image = torch.rand(1, 2, 11, 17, dtype=torch.float64)
    matrix = torch.tensor([[[1., .03, 1.2], [-.02, 1., .4], [.003, -.002, 1.]]], dtype=image.dtype)
    a, mask = warp(image, matrix)
    b, mask_b = warp(image, -matrix)
    torch.testing.assert_close(a, b)
    assert torch.equal(mask, mask_b) and not mask.all()
    with pytest.raises(ValueError, match='Singular'):
        inverse_homography(torch.zeros(1, 3, 3))


def test_cycle_photo_has_geometry_gradient_and_masks_boundary():
    image = torch.rand(1, 3, 15, 19, dtype=torch.float64)
    matrix = torch.tensor([[[[1., 0, .3], [0, 1., .2], [0, 0, 1.]]]],
                          dtype=image.dtype, requires_grad=True)
    reconstructed, masks = synthetic_view_reconstructions(image, matrix)
    loss = LAGRLoss.photometric_loss(image, reconstructed, masks)
    gradient, = torch.autograd.grad(loss, matrix)
    assert loss > 0 and gradient.abs().sum() > 0 and torch.isfinite(gradient).all()
    assert not masks[0].all()
    with pytest.raises(ValueError, match='detached'):
        LAGRLoss.photometric_loss(image, [reconstructed[0].detach()], masks)


def test_group_loss_ignores_warp_padding():
    q = torch.ones(1, 1, 15, 19)
    matrix = torch.tensor([[[1., 0, 3.], [0, 1., 2.], [0, 0, 1.]]])
    transformed, valid = warp(q, matrix)
    assert LAGRLoss.group_loss(q, transformed, matrix, valid).item() < 1e-6


def test_homography_resize_preserves_identity_and_translation_scale():
    matrix = torch.eye(3)[None]
    torch.testing.assert_close(resize_homography(matrix, (8, 10), (32, 40)), matrix)
    matrix[:, 0, 2] = 1
    transformed = resize_homography(matrix, (8, 10), (32, 40))
    assert transformed[0, 0, 2] == 4
