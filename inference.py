"""Uncropped inverse-depth inference; PNGs are visualization only."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from configs.paper import PaperConfig, CHECKPOINT_FORMAT_VERSION
from datasets.relative_depth import prepare_image
from model import LAGRNet


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION:
        raise ValueError('Expected learned RCL Fuse and the patch-intersection sheaf graph (format 3); earlier architectures are incompatible')
    config = PaperConfig(**checkpoint['config'])
    kwargs = config.model_kwargs()
    kwargs['pretrained_backbone'] = False
    model = LAGRNet(**kwargs).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval(), config


@torch.no_grad()
def predict(model, image, original_hw):
    prediction = model(image)['inverse_depth']
    return F.interpolate(prediction, size=original_hw, mode='bilinear', align_corners=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image')
    group.add_argument('--image-dir')
    parser.add_argument('--output-dir', default='outputs/inverse_depth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    paths = [Path(args.image)] if args.image else sorted(p for p in Path(args.image_dir).iterdir()
                                                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not paths:
        raise ValueError('No input images')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image, hw = prepare_image(path, config.short_side)
        q = predict(model, image[None].to(device), hw)[0, 0].cpu().numpy()
        # Include original extension to avoid collisions between x.png and x.jpg.
        np.save(output / f'{path.name}.inverse_depth.npy', q)
        span = float(q.max()-q.min())
        display = np.zeros_like(q) if span == 0 else (q-q.min())/span
        Image.fromarray((display*255).round().astype('uint8')).save(output / f'{path.name}.preview.png')
        print(f'{path.name}: saved raw relative inverse depth {tuple(q.shape)}')


if __name__ == '__main__':
    main()
