"""
Training script for LAGRNet on KITTI depth estimation.

Usage:
    python train.py --data_root /path/to/kitti --epochs 50 --batch_size 8
    python train.py --data_root /path/to/kitti --resume checkpoints/latest.pth
"""

import os
import time
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import LAGRNet, LAGRLoss
from datasets.kitti_dataset import get_kitti_dataloaders
from configs.kitti_config import KITTIConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train LAGRNet on KITTI')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of KITTI dataset')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--max_depth', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained backbone')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    return parser.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ====================== Depth Metrics ======================
def compute_depth_metrics(pred, gt, valid_mask, max_depth=80.0):
    """
    Compute standard KITTI depth evaluation metrics.

    Args:
        pred: predicted depth (B, 1, H, W)
        gt: ground truth depth (B, 1, H, W)
        valid_mask: validity mask (B, 1, H, W)
        max_depth: maximum depth for evaluation

    Returns:
        dict of metric names to values
    """
    # Flatten valid pixels
    mask = (valid_mask > 0.5) & (gt > 1e-3) & (gt < max_depth)
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if pred_valid.numel() == 0:
        return {k: 0.0 for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                                   'a1', 'a2', 'a3']}

    # Scale prediction to match GT range (since model outputs sigmoid in [0,1])
    # Convert model output from [0,1] to depth in meters
    pred_valid = pred_valid * max_depth
    pred_valid = pred_valid.clamp(min=1e-3, max=max_depth)

    thresh = torch.max(gt_valid / pred_valid, pred_valid / gt_valid)
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_rel = ((pred_valid - gt_valid).abs() / gt_valid).mean().item()
    sq_rel = (((pred_valid - gt_valid) ** 2) / gt_valid).mean().item()

    rmse = torch.sqrt(((pred_valid - gt_valid) ** 2).mean()).item()
    rmse_log = torch.sqrt(((torch.log(pred_valid) - torch.log(gt_valid)) ** 2).mean()).item()

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


# ====================== Training ======================
def train_one_epoch(model, criterion, train_loader, optimizer, scheduler,
                    device, epoch, cfg, logger, writer, global_step):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(train_loader):
        image = batch_data['image'].to(device)
        depth_gt = batch_data['depth'].to(device)
        valid_mask = batch_data['valid_mask'].to(device)

        # Forward pass
        outputs = model(image)

        # Construct loss batch dict
        loss_batch = {
            'I_ref': batch_data['image_raw'].to(device),
            'valid_mask': valid_mask,
            'N_patches': cfg.patch_grid[0] * cfg.patch_grid[1],
        }

        # Compute loss
        total_loss, loss_parts = criterion(outputs, loss_batch)

        # Supervised depth loss (scale-invariant L1)
        depth_pred = outputs['depth']
        # Scale prediction to depth range for supervised loss
        depth_pred_scaled = depth_pred * cfg.max_depth
        supervised_loss = (
            (depth_pred_scaled - depth_gt).abs() * valid_mask
        ).sum() / valid_mask.sum().clamp(min=1)

        # Total loss combines self-supervised components + supervised depth
        final_loss = total_loss + supervised_loss

        # Backward pass
        optimizer.zero_grad()
        final_loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch_loss += final_loss.item()
        global_step += 1

        # Logging
        if batch_idx % cfg.log_every == 0:
            logger.info(
                f'Epoch [{epoch}/{cfg.epochs}] Step [{batch_idx}/{num_batches}] '
                f'Loss: {final_loss.item():.4f} '
                f'(depth: {supervised_loss.item():.4f}, '
                f'sheaf: {loss_parts["sheaf"].item():.4f}, '
                f'smooth: {loss_parts["sm"].item():.4f})'
            )
            writer.add_scalar('train/total_loss', final_loss.item(), global_step)
            writer.add_scalar('train/depth_loss', supervised_loss.item(), global_step)
            writer.add_scalar('train/sheaf_loss', loss_parts['sheaf'].item(), global_step)
            writer.add_scalar('train/smooth_loss', loss_parts['sm'].item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    if scheduler is not None:
        scheduler.step()

    avg_loss = epoch_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, device, cfg, logger, writer, epoch):
    model.eval()
    metrics_sum = {k: 0.0 for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                                      'a1', 'a2', 'a3']}
    num_batches = 0

    for batch_data in val_loader:
        image = batch_data['image'].to(device)
        depth_gt = batch_data['depth'].to(device)
        valid_mask = batch_data['valid_mask'].to(device)

        outputs = model(image)
        depth_pred = outputs['depth']

        metrics = compute_depth_metrics(depth_pred, depth_gt, valid_mask, cfg.max_depth)
        for k in metrics_sum:
            metrics_sum[k] += metrics[k]
        num_batches += 1

    # Average metrics
    for k in metrics_sum:
        metrics_sum[k] /= max(num_batches, 1)

    logger.info(
        f'Validation Epoch [{epoch}] '
        f'abs_rel: {metrics_sum["abs_rel"]:.4f} | '
        f'rmse: {metrics_sum["rmse"]:.3f} | '
        f'a1: {metrics_sum["a1"]:.4f} | '
        f'a2: {metrics_sum["a2"]:.4f} | '
        f'a3: {metrics_sum["a3"]:.4f}'
    )

    # TensorBoard
    for k, v in metrics_sum.items():
        writer.add_scalar(f'val/{k}', v, epoch)

    return metrics_sum


def build_scheduler(optimizer, cfg):
    """Build learning rate scheduler with warmup."""
    if cfg.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
        )
    elif cfg.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
    else:
        scheduler = None
    return scheduler


def main():
    args = parse_args()
    cfg = KITTIConfig()

    # Override config with command line args
    cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.height is not None:
        cfg.height = args.height
    if args.width is not None:
        cfg.width = args.width
    if args.max_depth is not None:
        cfg.max_depth = args.max_depth
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.no_pretrained:
        cfg.pretrained_backbone = False

    # Setup
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    logger = setup_logging(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Configuration:\n{cfg}')
    logger.info(f'Device: {device}')

    # Data
    logger.info('Loading KITTI dataset...')
    train_loader, val_loader = get_kitti_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        height=cfg.height,
        width=cfg.width,
        num_workers=cfg.num_workers,
        max_depth=cfg.max_depth,
        use_right=cfg.use_right,
    )
    logger.info(f'Train samples: {len(train_loader.dataset)}, '
                f'Val samples: {len(val_loader.dataset)}')

    # Model
    logger.info('Building LAGRNet model...')
    model = LAGRNet(
        unified_channels=cfg.unified_channels,
        K_orbit=cfg.K_orbit,
        D_grade=cfg.D_grade,
        sheaf_dim=cfg.sheaf_dim,
        patch_grid=cfg.patch_grid,
        pretrained_backbone=cfg.pretrained_backbone,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')

    # Loss
    criterion = LAGRLoss(
        lam_pho=cfg.lam_pho,
        lam_grp=cfg.lam_grp,
        lam_sheaf=cfg.lam_sheaf,
        lam_sm=cfg.lam_sm,
        gamma=cfg.gamma_sm,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = build_scheduler(optimizer, cfg)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_abs_rel = float('inf')

    if args.resume and os.path.isfile(args.resume):
        logger.info(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_abs_rel = ckpt.get('best_abs_rel', float('inf'))
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f'Resumed at epoch {start_epoch}, global_step {global_step}')

    # ==================== Training Loop ====================
    logger.info('Starting training...')
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Warmup: linear LR ramp
        if epoch < cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.learning_rate * warmup_factor

        avg_loss, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer,
            scheduler if epoch >= cfg.warmup_epochs else None,
            device, epoch, cfg, logger, writer, global_step
        )

        epoch_time = time.time() - t0
        logger.info(f'Epoch [{epoch}/{cfg.epochs}] completed in {epoch_time:.1f}s, '
                    f'avg_loss: {avg_loss:.4f}')

        # Validation
        if (epoch + 1) % cfg.val_every == 0:
            metrics = validate(model, val_loader, device, cfg, logger, writer, epoch)

            # Save best model
            if metrics['abs_rel'] < best_abs_rel:
                best_abs_rel = metrics['abs_rel']
                save_path = os.path.join(cfg.checkpoint_dir, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_abs_rel': best_abs_rel,
                    'metrics': metrics,
                }, save_path)
                logger.info(f'Saved best model (abs_rel={best_abs_rel:.4f}) to {save_path}')

        # Save periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            save_path = os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_abs_rel': best_abs_rel,
            }, save_path)

        # Always save latest
        save_path = os.path.join(cfg.checkpoint_dir, 'latest.pth')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_abs_rel': best_abs_rel,
        }, save_path)

    writer.close()
    logger.info(f'Training complete. Best abs_rel: {best_abs_rel:.4f}')


if __name__ == '__main__':
    main()
