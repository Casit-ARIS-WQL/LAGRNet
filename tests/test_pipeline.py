import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from configs.paper import PaperConfig
from datasets.relative_depth import RelativeDepthDataset, TRAIN_SOURCES, normalize_rgb
from engine import objective, polynomial_lr
from evaluation import inverse_depth_metrics, summarize_records
from losses import affine_invariant_depth_loss, LAGRLoss
from model import LAGRNet, SwinBackbone


class ExplicitTestBackbone(nn.Module):
    """Small injected fixture, never used as a production Swin fallback."""
    out_channels = (4, 8, 12)
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(3, 4, 4, 4), nn.Conv2d(4, 8, 2, 2), nn.Conv2d(8, 12, 2, 2)])
    def forward(self, x):
        result = []
        for layer in self.layers:
            x = layer(x).relu()
            result.append(x)
        return result


def small_model():
    return LAGRNet(unified_channels=14, sheaf_dim=4, patch_grid=(3, 3),
                   pretrained_backbone=False, backbone=ExplicitTestBackbone())


def test_all_five_losses_and_backbone_photo_gradient():
    model = small_model()
    image = torch.rand(2, 3, 96, 128)
    batch = {'image_raw': image, 'image': normalize_rgb(image),
             'inverse_depth': torch.rand(2, 1, 96, 128), 'valid_mask': torch.ones(2, 1, 96, 128, dtype=torch.bool)}
    total, parts, outputs = objective(model, LAGRLoss(), batch)
    assert set(parts) == {'depth', 'pho', 'grp', 'sheaf', 'sm', 'total'}
    assert all(torch.isfinite(x) and x > 0 for x in parts.values())
    gradient, = torch.autograd.grad(parts['pho'], model.backbone.layers[0].weight, retain_graph=True)
    assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all()
    torch.testing.assert_close(total, parts['depth']+parts['pho']+.5*parts['grp']+.1*parts['sheaf']+.01*parts['sm'])
    before = model.backbone.layers[0].weight.detach().clone()
    fusion_before = model.rcl_fuse0.fuse_logits.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.backbone.layers[0].weight)
    assert not torch.equal(fusion_before, model.rcl_fuse0.fuse_logits)
    with pytest.raises(KeyError):
        LAGRLoss()(outputs, batch)


def test_affine_depth_supervision_and_lr():
    target = torch.linspace(.1, 1, 20).reshape(1, 1, 4, 5)
    mask = torch.ones_like(target, dtype=torch.bool)
    assert affine_invariant_depth_loss(3*target+7, target, mask) < 2e-6
    assert polynomial_lr(0, 480000) == 1e-4
    assert polynomial_lr(480000, 480000) == 0
    assert 0 < polynomial_lr(100000, 480000) < 1e-4


def test_evaluation_is_inverse_depth_not_metric_depth():
    inverse = torch.linspace(.1, 1, 20).reshape(1, 4, 5)
    mask = torch.ones_like(inverse, dtype=torch.bool)
    metrics = inverse_depth_metrics(4*inverse+2, inverse.reciprocal(), mask)
    assert metrics['abs_rel'] < 1e-6 and metrics['a1'] == 1
    assert metrics['positive_fraction'] == 1
    with pytest.raises(ValueError, match='Constant'):
        inverse_depth_metrics(torch.ones_like(inverse), inverse.reciprocal(), mask)
    rows = [dict(metrics, dataset=name) for name in ('KITTI','NYUv2','ETH3D')]
    assert 'three_dataset_mean' in summarize_records(rows)
    assert 'three_dataset_mean' not in summarize_records(rows[:2])


def test_manifests_six_sources_and_benchmark_separation(tmp_path):
    records = []
    for i, source in enumerate(TRAIN_SOURCES):
        image = np.arange(24*40*3, dtype=np.uint8).reshape(24, 40, 3)
        Image.fromarray(image).save(tmp_path/f'{i}.png')
        np.save(tmp_path/f'{i}.npy', np.linspace(1, 4, 24*40, dtype=np.float32).reshape(24, 40))
        records.append(dict(dataset=source, split='train', image=f'{i}.png', depth=f'{i}.npy', depth_scale=1))
    manifest = tmp_path/'train.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    dataset = RelativeDepthDataset([manifest], training=True, short_side=16)
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample['image'].shape == (3, 16, 16)
    assert sample['inverse_depth'].min() >= 0 and sample['inverse_depth'].max() <= 1
    records[0].update(dataset='KITTI', split='eigen_test')
    manifest.write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match='six labeled'):
        RelativeDepthDataset([manifest], training=True)
    manifest.write_text(json.dumps(records[0]))
    sample = RelativeDepthDataset([manifest], training=False, short_side=16)[0]
    assert sample['image'].shape[-2:] == (16, 27)
    assert sample['depth'].shape[-2:] == (24, 40)


def test_real_swin_dynamic_shapes_and_photo_backbone_gradient():
    backbone = SwinBackbone(pretrained=False, img_size=(32, 48), model_name='swin_tiny_patch4_window7_224')
    # This is a real Swin fixture, not a substitute architecture.
    model = LAGRNet(unified_channels=14, sheaf_dim=4, patch_grid=(3, 3), backbone=backbone).eval()
    image = torch.rand(1, 3, 32, 48)
    batch = {'image_raw': image, 'image': normalize_rgb(image),
             'inverse_depth': torch.rand(1, 1, 32, 48), 'valid_mask': torch.ones(1, 1, 32, 48, dtype=torch.bool)}
    _, parts, _ = objective(model, LAGRLoss(), batch)
    parameter = next(backbone.parameters())
    gradient, = torch.autograd.grad(parts['pho'], parameter)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        prediction = model(torch.rand(1, 3, 37, 61))['inverse_depth']
    assert prediction.shape == (1, 1, 37, 61)


def test_nonpositive_aligned_predictions_are_never_silently_dropped():
    pred = torch.tensor([[[1., 2., 3.]]])
    gt_inverse = torch.tensor([[[.1, .1, 3.]]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    with pytest.raises(ValueError, match='Nonpositive'):
        inverse_depth_metrics(pred, gt_inverse.reciprocal(), mask)
    excluded = inverse_depth_metrics(pred, gt_inverse.reciprocal(), mask, 'exclude')
    clamped = inverse_depth_metrics(pred, gt_inverse.reciprocal(), mask, 'clamp')
    assert excluded['n_scored'] == 2 and excluded['n_valid'] == 3
    assert clamped['n_scored'] == 3
    assert excluded['positive_fraction'] == pytest.approx(2/3)


def test_figure2_four_rcl_blocks_each_have_one_input_and_seven_grades():
    from model import RCL
    model = small_model().eval()
    observed = {}
    handles = []
    def inspect(name):
        def hook(module, args, output):
            assert len(args) == 1 and isinstance(args[0], torch.Tensor)
            observed[name] = (args[0].shape, output.shape,
                              len(module.pyramid(args[0])))
        return hook
    for name, module in model.named_modules():
        if isinstance(module, RCL):
            handles.append(module.register_forward_hook(inspect(name)))
    with torch.no_grad():
        prediction = model(torch.rand(2, 3, 64, 96))['inverse_depth']
    for handle in handles:
        handle.remove()
    assert set(observed) == {'rcl1', 'rcl2', 'rcl_fuse1', 'rcl_fuse0'}
    assert all(v[2] == 7 for v in observed.values())
    assert observed['rcl1'][1][-2:] == (4,6)
    assert observed['rcl2'][1][-2:] == (2,3)
    assert observed['rcl_fuse1'][1][-2:] == (4,6)
    assert observed['rcl_fuse0'][1][-2:] == (16,24)
    assert prediction.shape == (2,1,64,96)
    # Parameter sharing is internal to each module, not across four RCL blocks.
    blocks = [x for x in model.modules() if isinstance(x, RCL)]
    assert len({id(x.W_RCL) for x in blocks}) == 4
    assert len({id(x.fuse_logits) for x in blocks}) == 4


def test_inference_rejects_checkpoints_with_old_fusion_or_graph(tmp_path):
    from inference import load_model
    for version in (1, 2):
        checkpoint = tmp_path/f'old-{version}.pt'
        torch.save({'format_version': version}, checkpoint)
        with pytest.raises(ValueError, match='format 3'):
            load_model(checkpoint, torch.device('cpu'))
