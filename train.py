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
    """
    Wraps a Swin-T backbone and returns features of the first three stages.
    Uses timm if available; otherwise falls back to a lightweight conv pyramid
    that produces tensors with the exact same shapes/strides {4,8,16}.
    """
    # Swin patch_size=4, window_size=7 -> feature map must be divisible by 7
    _ALIGN = 28  # patch_size(4) * window_size(7)

    def __init__(self, pretrained: bool = True, img_size=(518, 518)):
        super().__init__()
        self.use_timm = False
        # Pad img_size to nearest multiple of _ALIGN so Swin window-attention works
        ph = math.ceil(img_size[0] / self._ALIGN) * self._ALIGN
        pw = math.ceil(img_size[1] / self._ALIGN) * self._ALIGN
        self._pad_h = ph - img_size[0]
        self._pad_w = pw - img_size[1]
        try:
            import timm
            # features_only=True with out_indices selecting the first 3 stages.
            # img_size must match the padded input so Swin's patch-embed assertion
            # and window-partition both succeed.
            self.model = timm.create_model(
                'swin_base_patch4_window7_224',
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2),
                img_size=(ph, pw),
            )
            self.out_channels = self.model.feature_info.channels()  # e.g. [128,256,512]
            self.use_timm = True
        except Exception:
            # Fallback pyramid: stride 4 -> 8 -> 16
            self.out_channels = [128, 256, 512]
            self.stem = nn.Sequential(
                nn.Conv2d(3, 128, 4, stride=4), nn.GELU(), nn.BatchNorm2d(128)
            )                                                        # stride 4
            self.down1 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(256)
            )                                                        # stride 8
            self.down2 = nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(512)
            )                                                        # stride 16

    def forward(self, I):
        if self.use_timm:
            B, C, H, W = I.shape
            # Pad to window-compatible size if necessary
            if self._pad_h > 0 or self._pad_w > 0:
                I = F.pad(I, (0, self._pad_w, 0, self._pad_h))
            feats = self.model(I)
            # timm Swin may return NHWC; convert to NCHW when last dim == channels
            out = []
            stride = 4
            for f in feats:
                if f.dim() == 4 and f.shape[-1] in self.out_channels:
                    f = f.permute(0, 3, 1, 2).contiguous()
                # Crop back to feature size corresponding to original H, W
                f = f[..., :H // stride, :W // stride]
                out.append(f)
                stride *= 2
            return out  # [F0, F1, F2]
        f0 = self.stem(I)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        return [f0, f1, f2]


# =============================================================================
# 2. GFM : Group-defined Feature Manifold                     [Eq. (2)-(4)]
#    - generate K matrices theta_{b,j} in PGL(3) with unit Frobenius norm  Eq.(2)
#    - warp F_(0) by theta^{-1} via bilinear interpolation                 Eq.(3)
#    - attention-weighted orbit aggregation + linear projection            Eq.(4)
# =============================================================================
class GFM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 8, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.eps = eps

        # Conditional generator: mu_b -> {theta_{b,j}}  (9*K params -> K 3x3 matrices)
        hidden = max(128, in_channels)
        self.W1 = nn.Linear(in_channels, hidden)      # W1 mu + b1
        self.W2 = nn.Linear(hidden, 9 * K)            # W2 ReLU(...) + b2   -> Eq.(2)

        # View-selection / attention head producing alpha in simplex Delta^{K-1}  Eq.(4)
        self.attn = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, K),
        )

        # Linear projection W_proj                                          Eq.(4)
        self.W_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    @staticmethod
    def _phi_normalize(M, eps):
        """Canonical mapping Phi: unit Frobenius norm.  Eq.(2):  M / (||M||_F + eps)."""
        # M: (B, K, 3, 3)
        norm = torch.linalg.matrix_norm(M, ord='fro').unsqueeze(-1).unsqueeze(-1)  # (B,K,1,1)
        return M / (norm + eps)

    def _make_base_grid(self, H, W, device, dtype):
        """Homogeneous target coordinates iota(p)=[u,v,1]^T over grid Omega."""
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')          # (H,W)
        ones = torch.ones_like(gx)
        # stack as [u(=x), v(=y), 1] ; use (x,y,1) convention consistent with iota
        grid = torch.stack([gx, gy, ones], dim=-1)              # (H,W,3)
        return grid

    def forward(self, F0):
        B, C, H, W = F0.shape
        device, dtype = F0.device, F0.dtype

        # ---- global context vector  mu_b = GlobalAvgPool(F_(0)) ----
        mu = F.adaptive_avg_pool2d(F0, 1).flatten(1)            # (B, C)

        # ---- Eq.(2): theta_{b,j} = Phi( W2 ReLU(W1 mu + b1) + b2 ) ----
        theta = self.W2(F.relu(self.W1(mu)))                   # (B, 9K)
        theta = theta.view(B, self.K, 3, 3)                    # (B,K,3,3)
        # bias toward identity for stable initialization
        theta = theta + torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        theta = self._phi_normalize(theta, self.eps)          # unit Frobenius norm, PGL(3) rep

        # ---- inverse transforms: theta^{-1}  (target -> source frame) ----
        theta_inv = torch.linalg.inv(theta)                   # (B,K,3,3)

        # ---- base homogeneous grid iota(p) ----
        base = self._make_base_grid(H, W, device, dtype)       # (H,W,3)
        base = base.view(1, 1, H * W, 3)                       # (1,1,HW,3)

        # tilde p' ~ theta^{-1} . iota(p)                       Eq.(3)
        # (B,K,3,3) @ (.,.,3,HW) -> (B,K,3,HW)
        src = torch.matmul(theta_inv, base.transpose(-1, -2).expand(B, self.K, 3, H * W))
        # dehomogenize pi(x1,x2,x3) = (x1/x3, x2/x3)
        denom = src[:, :, 2:3, :].clamp(min=self.eps)          # avoid /0
        u = src[:, :, 0, :] / denom.squeeze(2)                 # (B,K,HW)
        v = src[:, :, 1, :] / denom.squeeze(2)                 # (B,K,HW)

        # normalize to [-1,1] for grid_sample (bilinear interpolation -> Eq.(3))
        u_n = 2.0 * u / max(W - 1, 1) - 1.0
        v_n = 2.0 * v / max(H - 1, 1) - 1.0
        grid = torch.stack([u_n, v_n], dim=-1).view(B, self.K, H, W, 2)  # (B,K,H,W,2)

        # ---- warp F0 by each theta_j : eta^{(j)} = bilinear(F0, pi(tilde p')) ----
        F0_rep = F0.unsqueeze(1).expand(B, self.K, C, H, W).reshape(B * self.K, C, H, W)
        grid_r = grid.reshape(B * self.K, H, W, 2)
        eta = F.grid_sample(F0_rep, grid_r, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
        eta = eta.view(B, self.K, C, H, W)                    # (B,K,C,H,W)

        # ---- attention over orbit : alpha_b in Delta^{K-1} ----  Eq.(4)
        alpha = torch.softmax(self.attn(mu), dim=-1)          # (B,K)
        alpha = alpha.view(B, self.K, 1, 1, 1)

        # ---- aggregate : sum_j alpha_{b,j} eta^{(j)} , then W_proj ----  Eq.(4)
        agg = (alpha * eta).sum(dim=1)                        # (B,C,H,W)
        F_gfm = self.W_proj(agg)                              # (B,out,H,W)
        return F_gfm


# =============================================================================
# 3. RCL : Ring Convolution Layer (graded Cauchy product)        [Eq. RCL_eq]
#    W_ell = W_RCL ⊙ M_ell    (band-pass mask on output channels)
#    RCL_d(F) = sigma( BN( sum_{ell=0}^d U( (W_RCL⊙M_ell) * P_{d-ell}(F) ) ) )
# =============================================================================
class RCL(nn.Module):
    """
    Graded Ring Convolution.
    Inputs: a list of feature maps {F_0, ..., F_D} at degrees 0..D
            (degree d  <-> downsampling factor 2^d, i.e. coarser = higher degree).
    Output: list {RCL_0(F), ..., RCL_D(F)}.
    """
    def __init__(self, in_channels: int, out_channels: int, D: int, k: int = 3):
        super().__init__()
        self.D = D
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shared generator kernel W_RCL in R^{Cout x Cin x k x k}
        self.W_RCL = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        nn.init.kaiming_uniform_(self.W_RCL, a=math.sqrt(5))

        # channel-group size s = floor(Cout / (D+1))
        self.s = out_channels // (D + 1)

        # register disjoint band-pass masks M_ell over OUTPUT channels   [mask eq]
        masks = torch.zeros(D + 1, out_channels, 1, 1, 1)
        for ell in range(D + 1):
            lo = ell * self.s
            hi = (ell + 1) * self.s if ell < D else out_channels  # last group absorbs remainder
            masks[ell, lo:hi] = 1.0
        self.register_buffer('masks', masks)                  # (D+1, Cout,1,1,1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # 1x1 channel aligners so all input degrees share Cin (P_{d-ell} projection)
        # (assumes inputs already unified to in_channels; identity if so)
        self.proj = nn.ModuleList([nn.Identity() for _ in range(D + 1)])

    def _masked_kernel(self, ell):
        """W_ell = W_RCL ⊙ M_ell   (mask broadcast over Cin,k,k)."""
        m = self.masks[ell].view(self.out_channels, 1, 1, 1)   # (Cout,1,1,1)
        return self.W_RCL * m

    def forward(self, feats):
        """
        feats: list of length D+1; feats[d] is the degree-d feature map
               with spatial size H/2^d x W/2^d (coarser at higher d).
        """
        assert len(feats) == self.D + 1
        outputs = []
        for d in range(self.D + 1):
            target_size = feats[d].shape[-2:]                  # target scale d
            acc = 0.0
            # discrete Cauchy product: sum over ell + (d-ell) = d
            for ell in range(d + 1):
                src = self.proj[d - ell](feats[d - ell])       # P_{d-ell}(F)
                W_ell = self._masked_kernel(ell)               # W_RCL ⊙ M_ell
                conv = F.conv2d(src, W_ell, padding=self.k // 2)   # (W_ell) * P_{d-ell}(F)
                # U: bilinear interpolation from finer scale (d-ell) to target scale d
                if conv.shape[-2:] != target_size:
                    conv = F.interpolate(conv, size=target_size,
                                         mode='bilinear', align_corners=False)
                acc = acc + conv
            out = self.act(self.bn(acc))                       # sigma(BN(.))  -> RCL_d
            outputs.append(out)
        return outputs


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
    def __init__(self, in_channels: int, d: int, grid_hw=(7, 7),
                 tau: float = 0.5, n_steps: int = 2):
        super().__init__()
        self.d = d
        self.tau = tau
        self.n_steps = n_steps
        self.gh, self.gw = grid_hw
        self.N = self.gh * self.gw

        # lift features onto cover (0-cochain) via adaptive pooling -> V_b
        self.lift = nn.Conv2d(in_channels, d, kernel_size=1)

        # restriction-map generator Phi_theta: R^{2d} -> R^{d x d}  (then Cayley orthogonal)
        self.Phi = nn.Sequential(
            nn.Linear(2 * d, 4 * d), nn.ReLU(inplace=True),
            nn.Linear(4 * d, d * d),
        )
        self.cayley = CayleyOrthogonal(d)

        # per-step learnable channel mixing W_k  (k = 1, 2 for 2 Euler steps)
        self.Wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_steps)])
        self.act = nn.ELU()

        # build 1-skeleton adjacency of the patch grid (4-neighborhood)
        A = self._build_grid_adjacency(self.gh, self.gw)       # (N,N) binary
        self.register_buffer('A', A)
        # directed edge list (u, v) with fixed orientation u<v (for delta^0)
        edges = (A.triu(1) > 0).nonzero(as_tuple=False)        # (E,2): (u,v), u<v
        self.register_buffer('edge_index', edges.t().contiguous())  # (2,E)

    @staticmethod
    def _build_grid_adjacency(gh, gw):
        N = gh * gw
        A = torch.zeros(N, N)
        for r in range(gh):
            for c in range(gw):
                i = r * gw + c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < gh and 0 <= nc < gw:
                        j = nr * gw + nc
                        A[i, j] = 1.0
        return A

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
        Fv = self.cayley(raw_v.view(B, -1, self.d, self.d))     # (B,E,d,d) orthogonal
        Fu = self.cayley(raw_u.view(B, -1, self.d, self.d))     # (B,E,d,d) orthogonal
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

    def forward(self, feat):
        """
        feat: (B, Cin, Hf, Wf).  Returns:
          refined map (B, d, gh, gw), and sheaf energy loss term context (Fv,Fu,H).
        """
        B = feat.shape[0]
        # lift -> 0-cochain V_b : pool feature onto N patches arranged on gh x gw grid
        x = self.lift(feat)                                   # (B,d,Hf,Wf)
        x = F.adaptive_avg_pool2d(x, (self.gh, self.gw))     # (B,d,gh,gw)
        H = x.flatten(2).transpose(1, 2).contiguous()        # (B,N,d) = h^{(0)} = V_b

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
        return refined, energy


# =============================================================================
# 5. Decoder : vanilla up-sampling network
# =============================================================================
class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        chs = [in_channels, 256, 128, 64, 32]
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
        return torch.sigmoid(d)   # bounded disparity/depth in (0,1)


# =============================================================================
# 6. Full LAGRNet
# =============================================================================
class LAGRNet(nn.Module):
    def __init__(self,
                 unified_channels: int = 256,
                 K_orbit: int = 8,
                 D_grade: int = 1,         # RCL on the two coarser scales -> degrees {0,1}
                 sheaf_dim: int = 128,
                 patch_grid=(7, 7),
                 pretrained_backbone: bool = True,
                 img_size=(518, 518)):
        super().__init__()
        self.D = D_grade
        self.backbone = SwinBackbone(pretrained=pretrained_backbone, img_size=img_size)
        c0, c1, c2 = self.backbone.out_channels

        # ---- GFM on the finest map F_(0) ----   Eq.(2)-(4)
        self.gfm = GFM(in_channels=c0, out_channels=unified_channels, K=K_orbit)

        # ---- unify channels of F_(1), F_(2) before RCL ----
        self.unify1 = nn.Conv2d(c1, unified_channels, 1)
        self.unify2 = nn.Conv2d(c2, unified_channels, 1)

        # ---- RCL : graded fusion over the two coarser scales (degrees 0,1) ----
        # degree 0 <-> F_(1) (finer of the two), degree 1 <-> F_(2) (coarser)
        self.rcl = RCL(in_channels=unified_channels,
                       out_channels=unified_channels, D=D_grade, k=3)

        # ---- SM on the RCL-refined semantic feature (highest degree, coarsest) ----
        self.sm = SheafModule(in_channels=unified_channels, d=sheaf_dim,
                              grid_hw=patch_grid)

        # ---- fuse GFM (fine) + SM-refined semantic (coarse) for decoding ----
        self.fuse = nn.Conv2d(unified_channels + sheaf_dim, unified_channels, 1)
        self.decoder = UpsampleDecoder(unified_channels, out_channels=1)

    def forward(self, I):
        B, _, H, W = I.shape
        F0, F1, F2 = self.backbone(I)                         # Eq.(1)

        # GFM equivariant representation from finest scale
        F_gfm = self.gfm(F0)                                 # (B, U, H/4, W/4)

        # RCL graded fusion of the two coarser scales
        # feats[0] = degree-0 (finer, F_(1)), feats[1] = degree-1 (coarser, F_(2))
        u1 = self.unify1(F1)                                  # (B,U,H/8,W/8)
        u2 = self.unify2(F2)                                  # (B,U,H/16,W/16)
        rcl_out = self.rcl([u1, u2])                          # list len D+1; RCL_eq
        semantic = rcl_out[self.D]                            # coarsest RCL output -> SM

        # SM topological refinement on coarsest semantic feature
        sm_refined, sheaf_energy = self.sm(semantic)         # (B, sheaf_dim, gh, gw), (B,)

        # bring SM output and GFM output to a common spatial size, then fuse
        sm_up = F.interpolate(sm_refined, size=F_gfm.shape[-2:],
                              mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([F_gfm, sm_up], dim=1))  # (B,U,H/4,W/4)

        depth = self.decoder(fused, out_size=(H, W))         # (B,1,H,W)

        return {
            'depth': depth,
            'sheaf_energy': sheaf_energy,                    # for L_sheaf
            'F_gfm': F_gfm,
        }


# =============================================================================
# 7. Objective Losses
# =============================================================================
class LAGRLoss(nn.Module):
    """
    L = lam_pho L_pho + lam_grp L_grp + lam_sheaf L_sheaf + lam_sm L_sm
    """
    def __init__(self, lam_pho=1.0, lam_grp=0.1, lam_sheaf=0.1, lam_sm=0.1, gamma=1.0):
        super().__init__()
        self.lam_pho = lam_pho
        self.lam_grp = lam_grp
        self.lam_sheaf = lam_sheaf
        self.lam_sm = lam_sm
        self.gamma = gamma

    # ---- Photometric loss : L1 over valid region ----
    def photometric_loss(self, I_ref, I_synth_list, valid_mask=None):
        loss = 0.0
        S = len(I_synth_list)
        for I_s in I_synth_list:
            diff = (I_ref - I_s).abs()
            if valid_mask is not None:
                diff = diff * valid_mask
                loss = loss + diff.sum() / (valid_mask.sum().clamp(min=1) * I_ref.shape[1])
            else:
                loss = loss + diff.mean()
        return loss / max(S, 1)

    # ---- Group-consistency loss : ||D_pred - W_{g^-1}(D_pred')||_1 ----
    def group_loss(self, D_pred, D_pred_t, inv_grid):
        """
        D_pred   : depth from I            (B,1,H,W)
        D_pred_t : depth from I'=g(I)      (B,1,H,W)
        inv_grid : sampling grid realizing W_{g^-1} (B,H,W,2) in [-1,1]
        """
        D_warp = F.grid_sample(D_pred_t, inv_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        return (D_pred - D_warp).abs().mean()

    # ---- Sheaf energy loss : (1/BN) sum_b E(h_b^{(2)}) ----
    def sheaf_loss(self, sheaf_energy, N):
        # sheaf_energy: (B,)  already = sum_e 1/2 ||delta^0 h||^2 per sample
        B = sheaf_energy.shape[0]
        return sheaf_energy.sum() / (B * N)

    # ---- Edge-aware smoothness loss ----
    def smoothness_loss(self, D_pred, I):
        # gradients
        dDx = (D_pred[:, :, :, :-1] - D_pred[:, :, :, 1:]).abs()
        dDy = (D_pred[:, :, :-1, :] - D_pred[:, :, 1:, :]).abs()
        dIx = (I[:, :, :, :-1] - I[:, :, :, 1:]).abs().mean(1, keepdim=True)
        dIy = (I[:, :, :-1, :] - I[:, :, 1:, :]).abs().mean(1, keepdim=True)
        wx = torch.exp(-self.gamma * dIx)
        wy = torch.exp(-self.gamma * dIy)
        return (dDx * wx).mean() + (dDy * wy).mean()

    def forward(self, outputs, batch):
        """
        outputs: dict from LAGRNet.forward
        batch  : dict with keys for the various supervisions
        """
        D_pred = outputs['depth']
        losses = {}

        # photometric
        if 'I_ref' in batch and 'I_synth_list' in batch:
            losses['pho'] = self.photometric_loss(
                batch['I_ref'], batch['I_synth_list'], batch.get('valid_mask'))
        else:
            losses['pho'] = D_pred.new_zeros(())

        # group consistency
        if 'D_pred_t' in batch and 'inv_grid' in batch:
            losses['grp'] = self.group_loss(D_pred, batch['D_pred_t'], batch['inv_grid'])
        else:
            losses['grp'] = D_pred.new_zeros(())

        # sheaf energy
        N = batch.get('N_patches', outputs['sheaf_energy'].new_tensor(49)).item() \
            if isinstance(batch.get('N_patches', None), torch.Tensor) else batch.get('N_patches', 49)
        losses['sheaf'] = self.sheaf_loss(outputs['sheaf_energy'], N)

        # smoothness
        losses['sm'] = self.smoothness_loss(D_pred, batch['I_ref']) \
            if 'I_ref' in batch else D_pred.new_zeros(())

        total = (self.lam_pho * losses['pho']
                 + self.lam_grp * losses['grp']
                 + self.lam_sheaf * losses['sheaf']
                 + self.lam_sm * losses['sm'])
        losses['total'] = total
        return total, losses


# =============================================================================
# 8. Smoke test
# =============================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    model = LAGRNet(unified_channels=256, K_orbit=6, D_grade=1,
                    sheaf_dim=128, patch_grid=(7, 7),
                    pretrained_backbone=False)
    I = torch.randn(2, 3, 224, 224)
    out = model(I)
    print('depth       :', out['depth'].shape)        # (2,1,224,224)
    print('sheaf_energy:', out['sheaf_energy'].shape)  # (2,)

    crit = LAGRLoss()
    batch = {
        'I_ref': I,
        'I_synth_list': [I + 0.01 * torch.randn_like(I)],
        'N_patches': 49,
    }
    total, parts = crit(out, batch)
    print('total loss  :', float(total))
    print({k: float(v) for k, v in parts.items()})
