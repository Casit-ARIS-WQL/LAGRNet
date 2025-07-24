import math, torch, random
from torch.utils.data import DataLoader, Dataset
from model  import AlgebraicGeoNet
from losses import photometric_loss, sheaf_loss, group_loss, smoothness_loss

# — dummy dataset —
class DummySet(Dataset):
    def __init__(self, H=384, W=384, n=20):
        self.H,self.W,self.n = H,W,n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        ref = torch.rand(3,self.H,self.W)
        src = torch.rand(3,self.H,self.W)
        proj = lambda xy,d: 2*xy/self.W-1            
        return ref, [src], [proj], None

# — homo sampler —
def homo_sampler():
    ang = (random.random()*6-3)*math.pi/180
    tx,ty = (random.random()*0.05-0.025 for _ in range(2))
    Hm = torch.tensor([[ math.cos(ang),-math.sin(ang),tx],
                       [ math.sin(ang), math.cos(ang),ty],
                       [0,0,1]], dtype=torch.float32)
    return Hm


def dummy_collate(batch):
    refs  = torch.stack([b[0] for b in batch])          # (B,3,H,W)


    srcs  = [torch.stack([b[1][0] for b in batch])]     
    projs = [b[2][0] for b in batch]                    
    masks = None                                        

    return refs, srcs, projs, masks


# — train one step —
def train_step(model,opt,batch,device,lmbd=(1,0.5,0.2,0.1)):
    model.train()
    ref,src,proj,mask = batch
    ref = ref.to(device)
    src = [s.to(device) for s in src]
    if mask is not None: mask = mask.to(device)

    inv_d, delta = model(ref)

    l_photo  = photometric_loss(ref,inv_d,src,proj,mask)*lmbd[0]
    l_sheaf  = sheaf_loss(delta)*lmbd[1]
    l_group  = group_loss(model,ref,homo_sampler)*lmbd[2]
    l_smooth = smoothness_loss(inv_d,ref)*lmbd[3]
    loss = l_photo+l_sheaf+l_group+l_smooth

    opt.zero_grad(); loss.backward(); opt.step()
    return {k:v.item() for k,v in
            dict(total=loss,photo=l_photo,sheaf=l_sheaf,
                 group=l_group,smooth=l_smooth).items()}

# — main —
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = LAGRNet(pretrained=False).to(device)
    print("network:", net)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
    loader = DataLoader(DummySet(), batch_size=2,
                    shuffle=True, collate_fn=dummy_collate)
    # for epoch in range(2):
    #     for b in loader:
    #         # print(b)
    #         log = train_step(net,opt,b,device)
    #     print(f'epoch {epoch+1}:',log)
