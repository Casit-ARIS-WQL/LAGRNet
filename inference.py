"""
Inference script for LAGRNet depth estimation on KITTI.

Usage:
    # Single image inference
    python inference.py --checkpoint checkpoints/best.pth --image path/to/image.png

    # Batch inference on directory
    python inference.py --checkpoint checkpoints/best.pth --image_dir path/to/images/

    # Inference on KITTI val/test split
    python inference.py --checkpoint checkpoints/best.pth --data_root /path/to/kitti --split val

    # With colored depth visualization
    python inference.py --checkpoint checkpoints/best.pth --image path/to/image.png --colormap
"""

import os
import argparse
import time

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import LAGRNet
from configs.kitti_config import KITTIConfig


def parse_args():
    parser = argparse.ArgumentParser(description='LAGRNet Depth Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single input image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of input images for batch inference')
    parser.add_argument('--data_root', type=str, default=None,
                        help='KITTI data root for split-based inference')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='KITTI split for evaluation')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for predictions')
    parser.add_argument('--height', type=int, default=352)
    parser.add_argument('--width', type=int, default=1216)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--colormap', action='store_true',
                        help='Save colored depth visualizations')
    parser.add_argument('--save_numpy', action='store_true',
                        help='Save raw depth predictions as .npy files')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained LAGRNet model from checkpoint."""
    cfg = KITTIConfig()

    model = LAGRNet(
        unified_channels=cfg.unified_channels,
        K_orbit=cfg.K_orbit,
        D_grade=cfg.D_grade,
        sheaf_dim=cfg.sheaf_dim,
        patch_grid=cfg.patch_grid,
        pretrained_backbone=False,  # Not needed for inference
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    print(f'Model loaded from: {checkpoint_path}')
    if 'epoch' in ckpt:
        print(f'  Trained for {ckpt["epoch"] + 1} epochs')
    if 'best_abs_rel' in ckpt:
        print(f'  Best abs_rel: {ckpt["best_abs_rel"]:.4f}')

    return model


def preprocess_image(image_path, height, width):
    """
    Load and preprocess a single image for inference.

    Returns:
        input_tensor: (1, 3, H, W) normalized tensor
        original_size: (orig_H, orig_W)
        image_raw: original image as numpy array for visualization
    """
    image = Image.open(image_path).convert('RGB')
    original_size = (image.height, image.width)

    # Resize
    image_resized = TF.resize(image, [height, width],
                              interpolation=TF.InterpolationMode.BILINEAR)

    # To tensor and normalize
    image_tensor = TF.to_tensor(image_resized)  # (3, H, W)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_tensor)

    return image_normalized.unsqueeze(0), original_size, np.array(image)


def depth_to_colormap(depth, max_depth=80.0):
    """
    Convert depth map to a colored visualization using plasma colormap.

    Args:
        depth: (H, W) numpy array in meters
        max_depth: max depth for normalization

    Returns:
        colored: (H, W, 3) uint8 numpy array
    """
    try:
        import matplotlib.pyplot as plt
        depth_normalized = np.clip(depth / max_depth, 0, 1)
        # Invert so closer = warmer colors
        colored = plt.cm.plasma(1.0 - depth_normalized)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple grayscale
        depth_normalized = np.clip(depth / max_depth, 0, 1)
        gray = ((1.0 - depth_normalized) * 255).astype(np.uint8)
        colored = np.stack([gray, gray, gray], axis=-1)
    return colored


@torch.no_grad()
def predict_depth(model, image_tensor, device, max_depth=80.0):
    """
    Run inference on a single preprocessed image tensor.

    Args:
        model: LAGRNet model
        image_tensor: (1, 3, H, W) normalized tensor
        device: torch device
        max_depth: maximum depth in meters

    Returns:
        depth_map: (H, W) numpy array in meters
    """
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    depth_pred = outputs['depth']  # (1, 1, H, W) in [0, 1]

    # Convert from [0, 1] to meters
    depth_map = depth_pred.squeeze().cpu().numpy() * max_depth
    return depth_map


def infer_single_image(model, image_path, device, args):
    """Process a single image and save results."""
    print(f'Processing: {image_path}')

    # Preprocess
    input_tensor, original_size, image_raw = preprocess_image(
        image_path, args.height, args.width
    )

    # Predict
    t0 = time.time()
    depth_map = predict_depth(model, input_tensor, device, args.max_depth)
    inference_time = time.time() - t0
    print(f'  Inference time: {inference_time * 1000:.1f} ms')

    # Resize depth back to original size
    depth_resized = np.array(
        Image.fromarray(depth_map).resize(
            (original_size[1], original_size[0]), Image.BILINEAR
        )
    )

    # Save outputs
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Save as 16-bit PNG (KITTI format: depth * 256)
    depth_uint16 = (depth_resized * 256).astype(np.uint16)
    depth_png_path = os.path.join(args.output_dir, f'{basename}_depth.png')
    Image.fromarray(depth_uint16).save(depth_png_path)
    print(f'  Saved depth PNG: {depth_png_path}')

    # Save numpy
    if args.save_numpy:
        npy_path = os.path.join(args.output_dir, f'{basename}_depth.npy')
        np.save(npy_path, depth_resized)
        print(f'  Saved depth NPY: {npy_path}')

    # Save colormap visualization
    if args.colormap:
        colored = depth_to_colormap(depth_resized, args.max_depth)
        vis_path = os.path.join(args.output_dir, f'{basename}_depth_colored.png')
        Image.fromarray(colored).save(vis_path)
        print(f'  Saved colored depth: {vis_path}')

    return depth_resized


def infer_directory(model, image_dir, device, args):
    """Process all images in a directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    print(f'Found {len(image_files)} images in {image_dir}')

    total_time = 0
    for img_path in image_files:
        input_tensor, original_size, _ = preprocess_image(
            img_path, args.height, args.width
        )

        t0 = time.time()
        depth_map = predict_depth(model, input_tensor, device, args.max_depth)
        total_time += time.time() - t0

        # Resize and save
        depth_resized = np.array(
            Image.fromarray(depth_map).resize(
                (original_size[1], original_size[0]), Image.BILINEAR
            )
        )

        basename = os.path.splitext(os.path.basename(img_path))[0]

        depth_uint16 = (depth_resized * 256).astype(np.uint16)
        Image.fromarray(depth_uint16).save(
            os.path.join(args.output_dir, f'{basename}_depth.png')
        )

        if args.colormap:
            colored = depth_to_colormap(depth_resized, args.max_depth)
            Image.fromarray(colored).save(
                os.path.join(args.output_dir, f'{basename}_depth_colored.png')
            )

        if args.save_numpy:
            np.save(
                os.path.join(args.output_dir, f'{basename}_depth.npy'),
                depth_resized
            )

    avg_time = total_time / max(len(image_files), 1)
    print(f'\nProcessed {len(image_files)} images')
    print(f'Average inference time: {avg_time * 1000:.1f} ms ({1.0 / avg_time:.1f} FPS)')


def infer_kitti_split(model, data_root, split, device, args):
    """Run inference on a KITTI split and compute metrics if GT is available."""
    from datasets.kitti_dataset import KITTIDepthDataset

    dataset = KITTIDepthDataset(
        data_root=data_root,
        split=split,
        height=args.height,
        width=args.width,
        max_depth=args.max_depth,
        augment=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f'Running inference on KITTI {split} split ({len(dataset)} samples)...')

    # Metrics accumulation
    metrics_sum = {k: 0.0 for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                                      'a1', 'a2', 'a3']}
    num_valid = 0
    total_time = 0

    for idx, batch_data in enumerate(dataloader):
        image = batch_data['image'].to(device)
        depth_gt = batch_data['depth']
        valid_mask = batch_data['valid_mask']

        t0 = time.time()
        outputs = model(image)
        total_time += time.time() - t0

        depth_pred = outputs['depth'].cpu()  # (1, 1, H, W) in [0, 1]

        # Save prediction
        basename = f'{idx:06d}'
        depth_np = depth_pred.squeeze().numpy() * args.max_depth
        depth_uint16 = (depth_np * 256).astype(np.uint16)
        Image.fromarray(depth_uint16).save(
            os.path.join(args.output_dir, f'{basename}_depth.png')
        )

        if args.colormap:
            colored = depth_to_colormap(depth_np, args.max_depth)
            Image.fromarray(colored).save(
                os.path.join(args.output_dir, f'{basename}_depth_colored.png')
            )

        # Compute metrics if GT available
        if valid_mask.sum() > 0:
            from train import compute_depth_metrics
            metrics = compute_depth_metrics(
                depth_pred, depth_gt, valid_mask, args.max_depth
            )
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
            num_valid += 1

        if (idx + 1) % 100 == 0:
            print(f'  Processed {idx + 1}/{len(dataset)} images...')

    # Print results
    avg_time = total_time / max(len(dataset), 1)
    print(f'\n{"=" * 60}')
    print(f'Inference on KITTI {split} complete')
    print(f'Average inference time: {avg_time * 1000:.1f} ms ({1.0 / avg_time:.1f} FPS)')

    if num_valid > 0:
        print(f'\nDepth Evaluation Metrics (on {num_valid} valid samples):')
        print(f'{"=" * 60}')
        for k in metrics_sum:
            metrics_sum[k] /= num_valid
        print(f'  abs_rel : {metrics_sum["abs_rel"]:.4f}')
        print(f'  sq_rel  : {metrics_sum["sq_rel"]:.4f}')
        print(f'  rmse    : {metrics_sum["rmse"]:.3f}')
        print(f'  rmse_log: {metrics_sum["rmse_log"]:.4f}')
        print(f'  δ < 1.25    : {metrics_sum["a1"]:.4f}')
        print(f'  δ < 1.25^2  : {metrics_sum["a2"]:.4f}')
        print(f'  δ < 1.25^3  : {metrics_sum["a3"]:.4f}')
    print(f'{"=" * 60}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.checkpoint, device)

    # Run inference based on input mode
    if args.image is not None:
        infer_single_image(model, args.image, device, args)
    elif args.image_dir is not None:
        infer_directory(model, args.image_dir, device, args)
    elif args.data_root is not None:
        infer_kitti_split(model, args.data_root, args.split, device, args)
    else:
        print('ERROR: Please specify --image, --image_dir, or --data_root')
        return


if __name__ == '__main__':
    main()
