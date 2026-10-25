import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Backbone : Swin Transformer, extract first 3 stages           [Eq. (1)]
#    F_(i) in R^{B x C_i x H/2^{i+2} x W/2^{i+2}},  i in {0,1,2}
#    strides = {4, 8, 16}
# =============================================================================
class SwinBackbone(nn.Module):
    """Actual Swin stages at strides 4/8/16; no silent substitute backbone."""
    def __init__(self, pretrained=True, img_size=(518, 518),
                 model_name='swin_base_patch4_window7_224'):
        super().__init__()
        import timm
        if not model_name.startswith('swin_'):
            raise ValueError('The manuscript backbone must be a Swin model')
        self.model = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2), img_size=img_size, strict_img_size=False,
        )
        self.out_channels = self.model.feature_info.channels()
        self.output_fmt = str(self.model.output_fmt)

    def forward(self, image):
        h, w = image.shape[-2:]
        # Patch embedding needs multiples of four; Swin pads its own windows.
        image = F.pad(image, (0, (-w) % 4, 0, (-h) % 4))
        features = self.model(image)
        if 'NHWC' in self.output_fmt:
            features = [x.permute(0, 3, 1, 2).contiguous() for x in features]
        return [x[..., :math.ceil(h/s), :math.ceil(w/s)]
                for x, s in zip(features, (4, 8, 16))]


# =============================================================================
# 2. GFM : Group-defined Feature Manifold                     [Eq. (2)-(4)]
#    - generate K matrices theta_{b,j} in PGL(3) with controlled Frobenius norm  Eq.(2)
#    - warp F_(0) by theta^{-1} via bilinear interpolation                 Eq.(3)
#    - attention-weighted orbit aggregation + linear projection            Eq.(4)
# =============================================================================
class GFM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 4, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.eps = eps

        # Conditional generator: mu_b -> {theta_{b,j}}  (9*K params -> K 3x3 matrices)
        hidden = max(128, in_channels)
        self.W1 = nn.Linear(in_channels, hidden)      # W1 mu + b1
        self.W2 = nn.Linear(hidden, 9 * K)            # W2 ReLU(...) + b2   -> Eq.(2)

        # Small, nonzero perturbations retain useful valid overlap initially.
        nn.init.normal_(self.W2.weight, std=1e-5)
        nn.init.zeros_(self.W2.bias)

        # View-selection / attention head producing alpha in simplex Delta^{K-1}  Eq.(4)
        self.attn = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, K),
        )

        # Linear projection W_proj                                          Eq.(4)
        self.W_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    @staticmethod
    def _phi_normalize(M, eps):
        """Representative scaling Phi (epsilon-regularized Frobenius normalization).  Eq.(2):  M / (||M||_F + eps)."""
        # M: (B, K, 3, 3)
        norm = torch.linalg.matrix_norm(M, ord='fro').unsqueeze(-1).unsqueeze(-1)  # (B,K,1,1)
        return M / (norm + eps)

    def forward(self, features, return_geometry=False):
        from geometry import warp
        batch = features.shape[0]
        mu = F.adaptive_avg_pool2d(features, 1).flatten(1)
        raw = self.W2(F.relu(self.W1(mu))).view(batch, self.K, 3, 3)
        matrix = raw + torch.eye(3, device=raw.device, dtype=raw.dtype)
        theta = self._phi_normalize(matrix, self.eps)
        # A singular matrix raises explicitly; Frobenius scaling is not a
        # mathematical invertibility or equivariance constraint.
        warped = [warp(features, t)[0] for t in theta.unbind(1)]
        alpha = self.attn(mu).softmax(-1)
        aggregate = sum(alpha[:, j, None, None, None] * x for j, x in enumerate(warped))
        result = self.W_proj(aggregate)
        if return_geometry:
            return result, theta, alpha
        return result


# =============================================================================
# 3. RCL : Ring Convolution Layer (graded Cauchy product)        [Eq. RCL_eq]
#    W_ell = W_RCL ⊙ M_ell    (disjoint output-channel masks, not frequency bands)
#    RCL_d(F) = sigma( BN( sum_{ell=0}^d U( (W_RCL⊙M_ell) * P_{d-ell}(F) ) ) )
# =============================================================================
class DegreePyramid(nn.Module):
    """Average-pool ONE input tensor into D+1 degree-indexed features.

    P_j uses a 2**j window and stride, ceil boundary handling, and means
    over actual image samples. P_0 is the original input. Every degree is
    formed directly from that same input, not another backbone stage.
    """
    def __init__(self, D):
        super().__init__()
        if D < 0:
            raise ValueError('D must be nonnegative')
        self.D = D

    def forward(self, feature):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise TypeError('RCL takes one B x C x H x W feature tensor')
        return [F.avg_pool2d(feature, kernel_size=2**j, stride=2**j,
                             ceil_mode=True, count_include_pad=False)
                for j in range(self.D + 1)]


class RCL(nn.Module):
    """Figure 4: one feature -> Avg Pool grades -> Cauchy action -> Fuse.

    D is the maximum algebraic degree, not a number of RCL layers.
    A single shared kernel tensor generates disjoint output-channel masks.
    Each degree output has BN/ReLU after its Cauchy sum. Fuse resizes all
    degree outputs to the requested grid and learns normalized fusion weights.
    Channel alignment, if needed, is applied before this module.
    """
    def __init__(self, in_channels: int, out_channels: int, D: int,
                 k: int = 3, output_stride: int = 1):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError('The graded endomorphism formulation requires common channels')
        if D < 0 or out_channels < D + 1 or k % 2 != 1:
            raise ValueError('Require D >= 0, C >= D+1, and an odd kernel size')
        if output_stride not in (1, 2):
            raise ValueError('Figure 2 uses output strides 1 (fusion) or 2 (encoder branches)')
        self.D, self.k = D, k
        self.in_channels, self.out_channels = in_channels, out_channels
        self.output_stride = output_stride
        self.pyramid = DegreePyramid(D)
        self.W_RCL = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        nn.init.kaiming_uniform_(self.W_RCL, a=math.sqrt(5))
        self.s = out_channels // (D + 1)
        masks = torch.zeros(D + 1, out_channels, 1, 1, 1)
        for ell in range(D + 1):
            lo = ell * self.s
            hi = (ell + 1) * self.s if ell < D else out_channels
            masks[ell, lo:hi] = 1.0
        self.register_buffer('masks', masks)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        # beta_d = softmax(a)_d. Zero logits initialize an equal-weight Fuse.
        self.fuse_logits = nn.Parameter(torch.zeros(D + 1))

    def _masked_kernel(self, ell):
        return self.W_RCL * self.masks[ell].view(self.out_channels, 1, 1, 1)

    def graded_action(self, features):
        """The linear graded-ring action on an already constructed cochain."""
        if len(features) != self.D + 1:
            raise ValueError('A graded cochain must have D+1 components')
        outputs = []
        for d in range(self.D + 1):
            target_size = features[d].shape[-2:]
            terms = []
            for ell in range(d + 1):
                value = F.conv2d(features[d-ell], self._masked_kernel(ell),
                                 padding=self.k // 2)
                if value.shape[-2:] != target_size:
                    value = F.interpolate(value, size=target_size, mode='bilinear',
                                          align_corners=False)
                terms.append(value)
            outputs.append(sum(terms))
        return outputs

    def forward(self, feature, linear_only=False, return_degrees=False):
        features = self.pyramid(feature)
        linear = self.graded_action(features)
        if linear_only:
            return linear
        degrees = [self.act(self.bn(value)) for value in linear]
        target = tuple(math.ceil(v / self.output_stride) for v in feature.shape[-2:])
        weights = self.fuse_logits.softmax(dim=0)
        fused = sum(weights[d] * F.interpolate(value, size=target, mode='bilinear',
                                               align_corners=False)
                    for d, value in enumerate(degrees))
        return (fused, degrees) if return_degrees else fused


# =============================================================================
# 4. SM : Sheaf-based Module (cellular sheaf diffusion)
#    - cover by patches -> 0-cochain V_b
#    - restriction maps F_{v<e} = Phi_theta(h_v,h_u), orthogonal (Cayley)
#    - coboundary delta^0, sheaf Laplacian L_S, energy E(h)
#    - 2 explicit Euler diffusion steps with channel mixing W_k
# =============================================================================
class CayleyOrthogonal(nn.Module):
    """Map a raw d x d matrix to an orthogonal one via Cayley transform:
       Q = (I - A)(I + A)^{-1},  A = skew(M).  Ensures Q^T Q = I_d."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        # M: (..., d, d)
        A = 0.5 * (M - M.transpose(-1, -2))                    # skew-symmetric
        I = self.I.expand_as(A)
        Q = torch.linalg.solve(I + A, I - A)                   # (I-A)(I+A)^{-1}
        return Q


class SheafModule(nn.Module):
    """Cellular sheaf on the actual intersections of overlapping patches.

    grid_hw bounds the number of patch positions per axis. The feature grid
    is partitioned into integer bins; each window extends its bin by one cell
    on either side, clipped to the image. Pooling is performed inside each
    window. Identical windows on very small grids are merged. The Cech nerve's
    1-skeleton includes every pair sharing at least one feature-grid cell.
    """
    def __init__(self, in_channels: int, d: int, grid_hw=(7, 7),
                 tau: float = 0.5, n_steps: int = 2):
        super().__init__()
        if len(grid_hw) != 2 or any(int(v) != v or v < 1 for v in grid_hw):
            raise ValueError('The patch grid requires two positive integer bounds')
        if d < 1 or tau <= 0 or n_steps < 1:
            raise ValueError('Require positive stalk dimension, step size, and step count')
        self.d = d
        self.tau = tau
        self.n_steps = n_steps
        self.grid_hw = tuple(int(v) for v in grid_hw)
        self.gh = self.gw = self.N = 0
        self._cover_hw = None
        self._windows = []

        # Project channels, then pool within each cover element to obtain V_b.
        self.lift = nn.Conv2d(in_channels, d, kernel_size=1)

        # restriction-map generator Phi_theta: R^{2d} -> R^{d x d}  (then Cayley orthogonal)
        self.Phi = nn.Sequential(
            nn.Linear(2 * d, 4 * d), nn.ReLU(inplace=True),
            nn.Linear(4 * d, d * d),
        )
        self.cayley = CayleyOrthogonal(d)

        # per-step learnable channel mixing W_k  (k = 1, 2 for 2 Euler steps)
        self.Wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_steps)])
        self.act = nn.ReLU(inplace=False)
        # Geometry is derived from the current resolution, not checkpoint state.
        self.register_buffer('patch_bounds', torch.empty(0, 4, dtype=torch.long), persistent=False)
        self.register_buffer('A', torch.empty(0, 0, dtype=torch.bool), persistent=False)
        self.register_buffer('edge_index', torch.empty(2, 0, dtype=torch.long), persistent=False)

    @staticmethod
    def _axis_windows(length, maximum):
        count = min(length, maximum)
        # Half-open intervals: touching boundaries alone are not intersections.
        windows = [(max(0, i * length // count - 1),
                    min(length, (i + 1) * length // count + 1))
                   for i in range(count)]
        return list(dict.fromkeys(windows))

    def _set_patch_cover(self, height, width):
        if height < 1 or width < 1:
            raise ValueError('Cannot cover an empty feature grid')
        if self._cover_hw == (height, width):
            return
        rows = self._axis_windows(height, self.grid_hw[0])
        cols = self._axis_windows(width, self.grid_hw[1])
        self.gh, self.gw = len(rows), len(cols)
        self._windows = [(y0, y1, x0, x1) for y0, y1 in rows for x0, x1 in cols]
        self.N = len(self._windows)
        bounds = torch.tensor(self._windows, dtype=torch.long, device=self.patch_bounds.device)
        y0, y1, x0, x1 = bounds.unbind(1)
        overlap_y = torch.maximum(y0[:, None], y0[None, :]) < torch.minimum(y1[:, None], y1[None, :])
        overlap_x = torch.maximum(x0[:, None], x0[None, :]) < torch.minimum(x1[:, None], x1[None, :])
        adjacency = overlap_y & overlap_x
        adjacency.fill_diagonal_(False)
        self.patch_bounds = bounds
        self.A = adjacency
        # One orientation of each undirected edge; no duplicate energy terms.
        self.edge_index = adjacency.triu(1).nonzero(as_tuple=False).t().contiguous()
        self._cover_hw = (height, width)

    def patch_cochain(self, feat):
        """Project and average each actual window, returning B x N x d."""
        self._set_patch_cover(*feat.shape[-2:])
        projected = self.lift(feat)
        return torch.stack([
            F.adaptive_avg_pool2d(projected[..., y0:y1, x0:x1], 1).flatten(1)
            for y0, y1, x0, x1 in self._windows
        ], dim=1)

    def _restriction_maps(self, H):
        """
        Predict per-edge, per-incidence orthogonal restriction maps.
        H: (B, N, d).  Returns F_v_e, F_u_e of shape (B, E, d, d).
        For oriented edge e=(u,v):
            F_{v<e} = Cayley(Phi(h_v, h_u)),  F_{u<e} = Cayley(Phi(h_u, h_v))
        """
        B = H.shape[0]
        u_idx, v_idx = self.edge_index[0], self.edge_index[1]   # (E,)
        h_u = H[:, u_idx, :]                                    # (B,E,d)
        h_v = H[:, v_idx, :]                                    # (B,E,d)

        raw_v = self.Phi(torch.cat([h_v, h_u], dim=-1))         # F_{v<e}=Phi(h_v,h_u)
        raw_u = self.Phi(torch.cat([h_u, h_v], dim=-1))         # F_{u<e}=Phi(h_u,h_v)
        edge_count = self.edge_index.shape[1]
        Fv = self.cayley(raw_v.reshape(B, edge_count, self.d, self.d))
        Fu = self.cayley(raw_u.reshape(B, edge_count, self.d, self.d))
        return Fv, Fu

    def _coboundary(self, H, Fv, Fu):
        """(delta^0 h)_e = F_{v<e} h_v - F_{u<e} h_u  in R^d. Returns (B,E,d)."""
        u_idx, v_idx = self.edge_index[0], self.edge_index[1]
        h_u = H[:, u_idx, :].unsqueeze(-1)                      # (B,E,d,1)
        h_v = H[:, v_idx, :].unsqueeze(-1)
        Fv_hv = torch.matmul(Fv, h_v).squeeze(-1)              # (B,E,d)
        Fu_hu = torch.matmul(Fu, h_u).squeeze(-1)
        return Fv_hv - Fu_hu                                   # (B,E,d)

    def _apply_laplacian(self, H, Fv, Fu):
        """
        Node-wise sheaf Laplacian:
        (L_S h)_v = sum_{u:(u,v)} F_{v<e}^T (F_{v<e} h_v - F_{u<e} h_u).
        Computed via scattering the per-edge discrepancy back to both endpoints.
        Returns (B, N, d).
        """
        B = H.shape[0]
        u_idx, v_idx = self.edge_index[0], self.edge_index[1]
        disc = self._coboundary(H, Fv, Fu)                     # (B,E,d) = (delta^0 h)_e
        disc = disc.unsqueeze(-1)                              # (B,E,d,1)

        # contribution to endpoint v: + F_{v<e}^T disc
        contrib_v = torch.matmul(Fv.transpose(-1, -2), disc).squeeze(-1)   # (B,E,d)
        # contribution to endpoint u: - F_{u<e}^T disc
        contrib_u = -torch.matmul(Fu.transpose(-1, -2), disc).squeeze(-1)  # (B,E,d)

        out = torch.zeros(B, self.N, self.d, device=H.device, dtype=H.dtype)
        out.index_add_(1, v_idx, contrib_v)
        out.index_add_(1, u_idx, contrib_u)
        return out                                            # (B,N,d) = L_S h

    def sheaf_energy(self, H, Fv, Fu):
        """E(h) = 1/2 sum_e ||(delta^0 h)_e||^2 ."""
        disc = self._coboundary(H, Fv, Fu)                    # (B,E,d)
        return 0.5 * disc.pow(2).sum(dim=(1, 2))              # (B,)

    def forward(self, feat, return_diagnostics=False):
        """
        feat: (B, Cin, Hf, Wf).  Returns:
          refined map (B, d, gh, gw), and sheaf energy loss term context (Fv,Fu,H).
        """
        B = feat.shape[0]
        H = self.patch_cochain(feat)                         # h^{(0)} = V_b

        # ----- 2 explicit Euler diffusion steps (k=0,1) -----
        for k in range(self.n_steps):
            Fv, Fu = self._restriction_maps(H)               # recompute L_S^{(k)} maps
            LSh = self._apply_laplacian(H, Fv, Fu)           # (L_S^{(k)} h^{(k)})_v
            mixed = self.Wk[k](LSh)                           # W_{k+1} (L_S h)_v
            H = self.act(H - self.tau * mixed)               # h^{(k+1)} update

        # final restriction maps (for L_S^{(2)}) used in sheaf energy loss
        Fv, Fu = self._restriction_maps(H)
        energy = self.sheaf_energy(H, Fv, Fu)                # (B,)

        refined = H.transpose(1, 2).reshape(B, self.d, self.gh, self.gw)
        if return_diagnostics:
            u, v = self.edge_index
            graph = 0.5 * (H[:, v] - H[:, u]).square().sum((1, 2))
            return refined, energy, {'cochain': H, 'graph_energy': graph,
                                     'restriction_v': Fv, 'restriction_u': Fu,
                                     'patch_bounds': self.patch_bounds,
                                     'patch_adjacency': self.A,
                                     'patch_edges': self.edge_index}
        return refined, energy


# =============================================================================
# 5. Decoder : vanilla up-sampling network
# =============================================================================
class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        chs = [in_channels, 128, 32]  # stride 4 -> 2 -> 1
        blocks = []
        for i in range(len(chs) - 1):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(chs[-1], out_channels, 3, padding=1)

    def forward(self, x, out_size=None):
        for blk in self.blocks:
            x = blk(x)
        d = self.head(x)
        if out_size is not None:
            d = F.interpolate(d, size=out_size, mode='bilinear', align_corners=False)
        return d  # raw relative inverse depth; no metric scaling or min-max


# =============================================================================
# 6. Full LAGRNet
# =============================================================================
class LAGRNet(nn.Module):
    """Figure 2: two encoder RCLs and two top-down fusion RCLs.

    Each RCL receives exactly one tensor and internally average-pools its
    seven grades for D=6. The four RCL blocks have separate parameters;
    kernel sharing is within each block across its degree-indexed action.
    """
    def __init__(self, unified_channels=None, K_orbit=4, D_grade=6,
                 sheaf_dim=None, patch_grid=(7, 7), pretrained_backbone=True,
                 img_size=(518, 518), backbone_name='swin_base_patch4_window7_224',
                 sheaf_tau=0.5, backbone=None):
        super().__init__()
        self.D = D_grade
        self.backbone = backbone if backbone is not None else SwinBackbone(
            pretrained=pretrained_backbone, img_size=img_size, model_name=backbone_name)
        c0, c1, c2 = self.backbone.out_channels
        # Optional explicit width override for small test fixtures/ablations.
        n0, n1, n2 = (c0, c1, c2) if unified_channels is None else (unified_channels,) * 3
        sheaf_dim = c0 if sheaf_dim is None else sheaf_dim
        self.gfm = GFM(c0, n0, K=K_orbit)
        self.align1 = nn.Identity() if c1 == n1 else nn.Conv2d(c1, n1, 1)
        self.align2 = nn.Identity() if c2 == n2 else nn.Conv2d(c2, n2, 1)
        self.rcl1 = RCL(n1, n1, D_grade, output_stride=2)
        self.rcl2 = RCL(n2, n2, D_grade, output_stride=2)
        self.sm = SheafModule(n2, sheaf_dim, grid_hw=patch_grid, tau=sheaf_tau)
        self.align_fuse1 = nn.Conv2d(n1 + sheaf_dim, n1, 1)
        self.rcl_fuse1 = RCL(n1, n1, D_grade)
        self.align_fuse0 = nn.Conv2d(n0 + n1, n0, 1)
        self.rcl_fuse0 = RCL(n0, n0, D_grade)
        self.decoder = UpsampleDecoder(n0)

    def forward(self, image, diagnostics=False):
        from geometry import resize_homography
        f0, f1, f2 = self.backbone(image)
        fine, theta, alpha = self.gfm(f0, return_geometry=True)
        middle = self.rcl1(self.align1(f1))
        coarse = self.rcl2(self.align2(f2))
        sm_result = self.sm(coarse, return_diagnostics=diagnostics)
        refined, energy = sm_result[:2]
        up = F.interpolate(refined, size=middle.shape[-2:], mode='bilinear', align_corners=False)
        middle = self.rcl_fuse1(self.align_fuse1(torch.cat((middle, up), 1)))
        up = F.interpolate(middle, size=fine.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.rcl_fuse0(self.align_fuse0(torch.cat((fine, up), 1)))
        q = self.decoder(fused, out_size=image.shape[-2:])
        result = {'inverse_depth': q, 'sheaf_energy': energy, 'N_patches': self.sm.N,
                  'F_gfm': fine, 'attention': alpha,
                  'view_homographies': resize_homography(theta, f0.shape[-2:], image.shape[-2:])}
        if diagnostics:
            result.update(sm_result[2])
        return result


# Keep the public import path while losses live in their own testable module.
from losses import LAGRLoss  # noqa: E402
