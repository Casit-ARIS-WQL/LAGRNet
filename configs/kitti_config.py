"""
Configuration for KITTI depth estimation training with LAGRNet.
"""


class KITTIConfig:
    """Training configuration for KITTI depth estimation."""

    # ========================= Data =========================
    data_root: str = '/data/kitti'          # Root directory of KITTI dataset
    height: int = 518                       # Input image height
    width: int = 518                        # Input image width
    max_depth: float = 80.0                 # Maximum depth in meters
    min_depth: float = 1e-3                 # Minimum valid depth in meters
    use_right: bool = False                 # Use right camera images

    # ========================= Model =========================
    unified_channels: int = 256             # Unified feature channels
    K_orbit: int = 8                        # GFM orbit size
    D_grade: int = 1                        # RCL grading degree
    sheaf_dim: int = 128                    # Sheaf module latent dim
    patch_grid: tuple = (7, 7)             # Sheaf patch grid size
    pretrained_backbone: bool = True        # Use pretrained Swin backbone

    # ========================= Training =========================
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = 'cosine'           # 'cosine' or 'step'
    lr_step_size: int = 15                  # For step scheduler
    lr_gamma: float = 0.5                   # For step scheduler
    warmup_epochs: int = 3                  # Linear warmup epochs
    grad_clip: float = 1.0                  # Gradient clipping max norm

    # ========================= Loss =========================
    lam_pho: float = 1.0                    # Photometric loss weight
    lam_grp: float = 0.1                    # Group consistency loss weight
    lam_sheaf: float = 0.01                 # Sheaf energy loss weight
    lam_sm: float = 0.1                     # Smoothness loss weight
    gamma_sm: float = 1.0                   # Smoothness edge-awareness

    # ========================= Logging =========================
    log_dir: str = './runs'                 # TensorBoard log directory
    checkpoint_dir: str = './checkpoints'   # Checkpoint save directory
    save_every: int = 5                     # Save checkpoint every N epochs
    log_every: int = 50                     # Log metrics every N steps
    val_every: int = 1                      # Validate every N epochs

    # ========================= Inference =========================
    checkpoint_path: str = ''               # Path to trained checkpoint
    output_dir: str = './output'            # Inference output directory

    def __repr__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items()
                 if not k.startswith('_') and not callable(v)}
        return '\n'.join(f'{k}: {v}' for k, v in sorted(attrs.items()))
