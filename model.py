import math, torch, torch.nn as nn, torch.nn.functional as F

# ─── optional torch_geometric ──────────────────────────
try:
    from torch_geometric.nn import GCNConv
    HAVE_TG = True
except ImportError:
    HAVE_TG = False

# ─── GFM ───────────────────────────────────────────────
class GFM(nn.Module):
    def __init__(self, c: int, k: int = 8):
        super().__init__()
        self.k = k
        self.theta = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(c, 32), nn.ReLU(True), nn.Linear(32, 9*k))
        self.alpha = nn.Linear(c, k)
        self.proj  = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        B,C,H,W = x.shape
        thetas = self.theta(x).view(B, self.k, 3, 3)
        alphas = F.softmax(self.alpha(x.mean([-1,-2])), -1)   # (B,k)
        warped = []
        for i in range(self.k):
            grid = F.affine_grid(thetas[:, i, :2], x.size(), align_corners=False)
            warped.append(F.grid_sample(x, grid, align_corners=False))
        warped = torch.stack(warped, 1)                       # (B,k,C,H,W)
        x = (alphas.view(B,self.k,1,1,1)*warped).sum(1)
        return self.proj(x)

# ─── Ring Convolution Layer ────────────────────────────
def ring_mask(c_out: int, deg: int):
    m = torch.zeros(deg+1, c_out, 1, 1, 1)
    step = c_out // (deg+1)
    for d in range(deg+1):
        m[d, d*step:(d+1)*step] = 1.
    return m

class RCL(nn.Module):
    def __init__(self, c_in: int, c_out: int, deg: int = 2, stride: int = 2, k: int = 3):
        super().__init__()
        self.D, self.stride = deg, stride
        self.W = nn.Parameter(torch.randn(c_out, c_in, k, k) * .02)
        self.register_buffer('mask', ring_mask(c_out, deg))
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        outs = []
        for d in range(self.D+1):
            x_d = F.avg_pool2d(x, 2**d) if d else x
            y_d = F.conv2d(x_d, self.W*self.mask[d], stride=self.stride,
                           padding=self.W.size(-1)//2)
            outs.append(y_d)
        tgt = outs[0].shape[-2:]
        outs = [y if y.shape[-2:] == tgt else
                F.interpolate(y, size=tgt, mode='bilinear', align_corners=False)
                for y in outs]
        y = sum(outs)
        return F.relu_(self.bn(y))

# ─── SBM ───────────────────────────────────────────────
class SBM(nn.Module):
    def __init__(self, c: int, patch: int = 6, stride: int = 3):
        super().__init__()
        self.p, self.s = patch, stride
        self.fallback = nn.Conv2d(c, c, 1, bias=False)
        if HAVE_TG:
            self.g1, self.g2 = GCNConv(c, c), GCNConv(c, c)

    def _edges(self, nH, nW, device):
        idx = torch.arange(nH*nW, device=device).view(nH,nW)
        e=[]
        for dh,dw in [(0,1),(1,0)]:
            s = idx[:nH-dh,:nW-dw].reshape(-1)
            t = idx[dh:    ,dw:   ].reshape(-1)
            e += [torch.stack([s,t]), torch.stack([t,s])]
        return torch.cat(e,1)

    def forward(self, x):
        B,C,H,W = x.shape
        if H < self.p or W < self.p or not HAVE_TG:
            delta_tau = torch.tensor(0., device=x.device, requires_grad=True)
            return self.fallback(x), delta_tau

        patches = x.unfold(2,self.p,self.s).unfold(3,self.p,self.s)
        nH,nW = patches.size(2), patches.size(3)
        tok   = patches.mean([-1,-2]).flatten(2).transpose(1,2)      # (B,N,C)
        edge  = self._edges(nH,nW,x.device)
        outs, deltas = [], []
        for b in range(B):
            f = tok[b]
            h1 = F.relu_(self.g1(f, edge))
            h2 = F.relu_(self.g2(h1, edge))
            outs.append(h2)
            deltas.append((h2 - h1).pow(2).mean())                
        out_feat = torch.stack(outs).view(B,nH,nW,C).permute(0,3,1,2)
        out_feat = F.interpolate(out_feat, (H,W), mode='bilinear', align_corners=False)
        delta_tau = torch.stack(deltas).mean()
        return out_feat, delta_tau

# ─── Up-RCL ────────────────────────────────────────────
class UpRCL(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, 2, 2)
        self.rcl= RCL(c_in+c_skip, c_out, deg=2, stride=1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.rcl(torch.cat([x,skip],1))


class AlgebraicGeoNet(nn.Module):
    def __init__(self, backbone='swin_base_patch4_window12_384', pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True, out_indices=(0,1,2,3))
        self.adj = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self._fix_backbone_channels()

        self.gfm  = GFM(96)
        self.rcl1 = RCL(192,192)
        self.rcl2 = RCL(384,384)
        self.sbm  = SBM(384)
        self.up1  = UpRCL(384,192,96)
        self.up2  = UpRCL(96, 96, 64)
        self.head = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(True),
                                  nn.Conv2d(64,1,1), nn.Softplus())
    
    def _lazy_adj(self, feats, device):
        target = [96, 192, 384]
        for i, (f, t) in enumerate(zip(feats, target)):
            in_c = f.size(1)
            layer = self.adj[i]
            need_rebuild = (
                isinstance(layer, nn.Conv2d) and layer.in_channels != in_c
            ) or (
                isinstance(layer, nn.Identity) and in_c != t
            )
            if need_rebuild:
                self.adj[i] = (
                    nn.Identity() if in_c == t
                    else nn.Conv2d(in_c, t, 1, bias=False)
                ).to(device)


    def _fix_backbone_channels(self):
        ch = self.backbone.feature_info.channels()
        target = [96,192,384]
        for i,(c,t) in enumerate(zip(ch[:3],target)):
            if c!=t:
                self.adj[i] = nn.Conv2d(c,t,1,bias=False)


    def _adapt_img_size(self, x):
        B, C, H, W = x.shape
        pe = self.backbone.patch_embed
        if (H, W) == tuple(pe.img_size):
            return

        gh, gw = H // pe.patch_size[0], W // pe.patch_size[1]
        pe.img_size = (H, W)
        pe.grid_size = (gh, gw)
        pe.num_patches = gh * gw

        base = self.backbone.model if hasattr(self.backbone, "model") else self.backbone
        if hasattr(base, "stages"):
            for stage in base.stages:
                for blk in stage.blocks:
                    blk.attn_mask = None


    def forward(self, x):
        self._adapt_img_size(x)

        f0,f1,f2,_ = self.backbone(x)
        self._lazy_adj((f0, f1, f2), x.device)
        f0,f1,f2 = self.adj[0](f0), self.adj[1](f1), self.adj[2](f2)

        g  = self.gfm(f0)
        r1 = self.rcl1(f1)
        r2 = self.rcl2(f2)
        s, delta_tau = self.sbm(r2)
        u1 = self.up1(s, r1)
        u2 = self.up2(u1, g)
        depth = F.interpolate(self.head(u2), size=x.shape[2:], mode='bilinear', align_corners=False)
        return depth, delta_tau         
