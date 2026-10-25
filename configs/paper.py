"""LAGRNet architecture, loss, training, and preprocessing configuration."""
from dataclasses import dataclass, asdict

CHECKPOINT_FORMAT_VERSION = 3  # learned RCL Fuse and patch-intersection sheaf graph


@dataclass
class PaperConfig:
    backbone_name: str = 'swin_base_patch4_window7_224'
    pretrained_backbone: bool = True
    unified_channels: int | None = None  # preserve the 128/256/512 channels in Figure 2
    K_orbit: int = 4
    D_grade: int = 6
    sheaf_dim: int | None = None  # C_(0), as in the manuscript
    patch_grid: tuple = (7, 7)  # maximum window positions per axis; one-cell overlap halo
    sheaf_tau: float = 0.5  # diffusion step size
    short_side: int = 518
    batch_size: int = 128  # GLOBAL, divided across distributed workers
    total_steps: int = 480000
    learning_rate: float = 1e-4
    poly_power: float = 0.9
    weight_decay: float = 1e-4  # AdamW weight decay
    lam_pho: float = 1.0
    lam_grp: float = 0.5
    lam_sheaf: float = 0.1
    lam_sm: float = 0.01
    gamma_sm: float = 1.0  # smoothness edge sensitivity
    augmentation_magnitude: float = 0.05  # perturbation magnitude in normalized coordinates
    jitter: float = 0.1  # brightness/contrast/saturation
    seed: int = 42
    num_workers: int = 4
    save_every: int = 5000
    log_every: int = 50

    def model_kwargs(self):
        names = ('backbone_name', 'pretrained_backbone', 'unified_channels',
                 'K_orbit', 'D_grade', 'sheaf_dim', 'patch_grid', 'sheaf_tau')
        result = {name: getattr(self, name) for name in names}
        result['img_size'] = (self.short_side, self.short_side)
        return result

    def to_dict(self):
        return asdict(self)
