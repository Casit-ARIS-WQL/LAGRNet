import math, torch, torch.nn.functional as F

# — photometric —
def photometric_loss(ref_img, ref_depth, src_imgs, proj_fns, mask=None):
    B,_,H,W = ref_img.shape
    yy,xx = torch.meshgrid(torch.arange(H,device=ref_img.device),
                           torch.arange(W,device=ref_img.device), indexing='ij')
    xy = torch.stack([xx,yy],-1).float().view(1,H,W,2).expand(B,-1,-1,-1)
    inv_d = 1.0 / (ref_depth.squeeze(1)+1e-6)
    losses=[]
    for src,proj in zip(src_imgs,proj_fns):
        grid = proj(xy, inv_d)                           
        warp = F.grid_sample(src, grid, align_corners=False)
        diff = (ref_img-warp).abs()
        if mask is not None: diff *= mask
        losses.append(diff.mean())
    return torch.stack(losses).mean()

# — sheaf —
def sheaf_loss(delta_tau):
    return (delta_tau**2).mean()

# — group —
def group_loss(model, img, homo_sampler):
    B = img.size(0)

    thetas = torch.stack([homo_sampler().to(img.device)[:2] for _ in range(B)], 0)  # (B,2,3)
    d1, _ = model(img)

    grid  = F.affine_grid(thetas, img.size(), align_corners=False)
    img2  = F.grid_sample(img, grid, align_corners=False)
    d2, _ = model(img2)
    inv_thetas = torch.inverse(torch.cat([thetas, torch.tensor([[[0,0,1]]]*B,
                                   device=img.device)], dim=1))[:, :2]  # (B,2,3)
    g_inv = F.affine_grid(inv_thetas, d2.size(), align_corners=False)
    d2_w  = F.grid_sample(d2, g_inv, align_corners=False)

    return (d1 - d2_w).abs().mean()


# — smooth —
def smoothness_loss(depth, img, alpha=10):
    dx = depth[..., :,1:] - depth[...,:,:-1]
    dy = depth[..., 1:,:] - depth[...,:-1,:]
    gx = img.mean(1,keepdim=True)[..., :,1:] - img.mean(1,keepdim=True)[...,:,:-1]
    gy = img.mean(1,keepdim=True)[..., 1:,:] - img.mean(1,keepdim=True)[...,:-1,:]
    wx = torch.exp(-alpha*gx.abs()).detach()
    wy = torch.exp(-alpha*gy.abs()).detach()
    return (dx.abs()*wx).mean() + (dy.abs()*wy).mean()
