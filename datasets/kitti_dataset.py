"""
KITTI Depth Estimation Dataset

Supports both the KITTI Depth Prediction benchmark (raw + annotated depth)
and the Eigen split commonly used for monocular depth estimation.

Expected directory structure:
    data_root/
    ├── raw_data/                     # KITTI raw synced+rectified data
    │   ├── 2011_09_26/
    │   │   ├── 2011_09_26_drive_0001_sync/
    │   │   │   ├── image_02/data/   # left color camera
    │   │   │   ├── image_03/data/   # right color camera
    │   │   │   └── ...
    │   │   └── calib_cam_to_cam.txt
    │   └── ...
    ├── depth_annotated/              # KITTI depth completion ground truth
    │   ├── train/
    │   │   ├── 2011_09_26_drive_0001_sync/
    │   │   │   └── proj_depth/groundtruth/image_02/
    │   │   └── ...
    │   └── val/
    └── splits/                       # Eigen or official split files
        ├── eigen_train_files.txt
        ├── eigen_val_files.txt
        └── eigen_test_files.txt
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class KITTIDepthDataset(Dataset):
    """
    KITTI dataset for monocular depth estimation.

    Each sample returns:
        - image: RGB tensor (3, H, W) normalized to [0, 1]
        - depth: ground truth depth map (1, H, W) in meters
        - valid_mask: boolean mask (1, H, W) indicating valid depth pixels

    Args:
        data_root: Root directory of KITTI data.
        split: One of 'train', 'val', 'test'.
        split_file: Path to split file (one sample per line). If None, will
                    look in data_root/splits/eigen_{split}_files.txt.
        height: Target image height (default 352 for KITTI).
        width: Target image width (default 1216 for KITTI).
        max_depth: Maximum depth value in meters (default 80.0 for KITTI).
        min_depth: Minimum valid depth in meters (default 1e-3).
        augment: Whether to apply data augmentation (training only).
        use_right: Also include right camera images (doubles dataset size).
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        split_file: str = None,
        height: int = 352,
        width: int = 1216,
        max_depth: float = 80.0,
        min_depth: float = 1e-3,
        augment: bool = False,
        use_right: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.height = height
        self.width = width
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.augment = augment and (split == 'train')
        self.use_right = use_right

        # Load file list
        if split_file is None:
            split_file = os.path.join(data_root, 'splits', f'eigen_{split}_files.txt')

        self.samples = self._load_split(split_file)

        # Image normalization (ImageNet stats for pretrained backbone)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _load_split(self, split_file):
        """
        Load split file. Each line format:
            date/drive_sync image_index l_or_r
        e.g.:
            2011_09_26/2011_09_26_drive_0001_sync 0000000000 l
        """
        samples = []
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Please provide a valid split file or place it at the expected path."
            )
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 3:
                    folder, frame_id, side = parts
                elif len(parts) == 2:
                    folder, frame_id = parts
                    side = 'l'
                else:
                    continue
                samples.append((folder, frame_id, side))

                if self.use_right and side == 'l':
                    samples.append((folder, frame_id, 'r'))

        return samples

    def _get_image_path(self, folder, frame_id, side):
        """Construct path to RGB image."""
        cam = 'image_02' if side == 'l' else 'image_03'
        return os.path.join(
            self.data_root, 'raw_data', folder, cam, 'data',
            f'{frame_id}.png'
        )

    def _get_depth_path(self, folder, frame_id, side):
        """Construct path to depth ground truth."""
        cam = 'image_02' if side == 'l' else 'image_03'
        # Try annotated depth first (from depth completion benchmark)
        sub_split = 'train' if self.split == 'train' else 'val'
        depth_path = os.path.join(
            self.data_root, 'depth_annotated', sub_split,
            folder.split('/')[-1], 'proj_depth', 'groundtruth',
            cam, f'{frame_id}.png'
        )
        if os.path.isfile(depth_path):
            return depth_path

        # Fallback: velodyne projected depth
        depth_path = os.path.join(
            self.data_root, 'depth_annotated', sub_split,
            folder.split('/')[-1], 'proj_depth', 'velodyne_raw',
            cam, f'{frame_id}.png'
        )
        return depth_path

    def _load_image(self, path):
        """Load RGB image as PIL Image."""
        img = Image.open(path).convert('RGB')
        return img

    def _load_depth(self, path):
        """
        Load KITTI depth map (16-bit PNG, depth in mm -> convert to meters).
        Invalid pixels have value 0.
        """
        if not os.path.isfile(path):
            return None
        depth = Image.open(path)
        depth = np.array(depth, dtype=np.float32)
        # KITTI stores depth in uint16 with factor 256
        depth = depth / 256.0
        return depth

    def _augment(self, image, depth):
        """Apply training augmentations."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            depth = np.fliplr(depth).copy()

        # Random brightness, contrast, saturation
        if torch.rand(1).item() > 0.5:
            brightness = 0.2 * torch.rand(1).item() + 0.9  # [0.9, 1.1]
            image = TF.adjust_brightness(image, brightness)

        if torch.rand(1).item() > 0.5:
            contrast = 0.2 * torch.rand(1).item() + 0.9
            image = TF.adjust_contrast(image, contrast)

        if torch.rand(1).item() > 0.5:
            saturation = 0.2 * torch.rand(1).item() + 0.9
            image = TF.adjust_saturation(image, saturation)

        return image, depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, frame_id, side = self.samples[idx]

        # Load image
        img_path = self._get_image_path(folder, frame_id, side)
        image = self._load_image(img_path)

        # Load depth
        depth_path = self._get_depth_path(folder, frame_id, side)
        depth = self._load_depth(depth_path)

        # Handle missing depth (e.g., test set)
        if depth is None:
            depth = np.zeros((image.height, image.width), dtype=np.float32)

        # Apply augmentation before resizing
        if self.augment:
            image, depth = self._augment(image, depth)

        # Resize
        image = TF.resize(image, [self.height, self.width], interpolation=TF.InterpolationMode.BILINEAR)
        depth = Image.fromarray(depth)
        depth = TF.resize(depth, [self.height, self.width], interpolation=TF.InterpolationMode.NEAREST)
        depth = np.array(depth, dtype=np.float32)

        # Convert to tensors
        image_tensor = TF.to_tensor(image)  # (3, H, W) in [0, 1]

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # Clamp depth to valid range
        valid_mask = (depth_tensor > self.min_depth) & (depth_tensor < self.max_depth)
        depth_tensor = depth_tensor.clamp(self.min_depth, self.max_depth)

        # Normalize image for model input
        image_normalized = self.normalize(image_tensor)

        return {
            'image': image_normalized,        # (3, H, W) normalized
            'image_raw': image_tensor,        # (3, H, W) [0, 1] for visualization
            'depth': depth_tensor,            # (1, H, W) meters
            'valid_mask': valid_mask.float(), # (1, H, W)
            'path': img_path,
        }


def get_kitti_dataloaders(
    data_root: str,
    batch_size: int = 8,
    height: int = 352,
    width: int = 1216,
    num_workers: int = 4,
    max_depth: float = 80.0,
    use_right: bool = False,
):
    """
    Convenience function to create train/val dataloaders.

    Returns:
        train_loader, val_loader
    """
    train_dataset = KITTIDepthDataset(
        data_root=data_root,
        split='train',
        height=height,
        width=width,
        max_depth=max_depth,
        augment=True,
        use_right=use_right,
    )

    val_dataset = KITTIDepthDataset(
        data_root=data_root,
        split='val',
        height=height,
        width=width,
        max_depth=max_depth,
        augment=False,
        use_right=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
