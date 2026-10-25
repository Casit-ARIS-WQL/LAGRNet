"""A complete differentiable training step shared by CLI and integration tests."""
from datasets.relative_depth import normalize_rgb
from geometry import random_homography, warp, synthetic_view_reconstructions


def objective(model, criterion, batch, augmentation_magnitude=0.05):
    outputs = model(batch['image'])
    inputs = dict(batch)
    if criterion.weights['pho']:
        inputs['reconstructions'], inputs['reconstruction_masks'] = synthetic_view_reconstructions(
            batch['image_raw'], outputs['view_homographies'])
    if criterion.weights['grp']:
        image = batch['image_raw']
        transform = random_homography(image.shape[0], image.shape[-2:], image.device,
                                      image.dtype, augmentation_magnitude)
        augmented, valid = warp(image, transform)
        transformed = model(normalize_rgb(augmented))
        inputs.update(transformed_prediction=transformed['inverse_depth'],
                      group_transform=transform, transformed_valid=valid)
    total, parts = criterion(outputs, inputs)
    return total, parts, outputs


def polynomial_lr(step, total_steps, initial_lr=1e-4, power=0.9):
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError('Require 0 <= step <= total_steps')
    return initial_lr * (1 - step / total_steps) ** power
