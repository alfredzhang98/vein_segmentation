"""Causal recurrence at 1/8 and 1/16; pretrained decoder and native resolution retained."""
import torch
from torch import nn
from torch.nn import functional as F
from models.unet.model import UNet

class ConvGRU(nn.Module):
    def __init__(self,features,hidden=64):
        super().__init__();self.hidden=hidden
        self.input=nn.Conv2d(features,hidden,1)
        self.gates=nn.Conv2d(hidden*2,hidden*2,3,padding=1)
        self.candidate=nn.Conv2d(hidden*2,hidden,3,padding=1)
        self.output=nn.Conv2d(hidden,features,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,h=None):
        z=self.input(x)
        if h is None:h=torch.zeros_like(z)
        reset,update=torch.sigmoid(self.gates(torch.cat((z,h),1))).chunk(2,1)
        candidate=torch.tanh(self.candidate(torch.cat((z,reset*h),1)))
        h=(1-update)*h+update*candidate
        return x+self.output(h),h

class TrackingUNet(nn.Module):
    def __init__(self,unet_config,hidden=64,max_flow=24):
        super().__init__();c=int(unet_config.get('base_ch',32));bilinear=unet_config.get('bilinear',False)
        self.config=dict(unet_config=unet_config,hidden=hidden,max_flow=max_flow)
        self.unet=UNet(1,3,bilinear,c,unet_config.get('dropout',0.3))
        self.gru8=ConvGRU(c*8,hidden);self.gru16=ConvGRU(c*16//(2 if bilinear else 1),hidden)
        self.transition=nn.Sequential(nn.Conv2d(c+3,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,5,1))
        nn.init.zeros_(self.transition[-1].weight);nn.init.zeros_(self.transition[-1].bias)
        self.max_flow=max_flow
    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.unet.parameters()):
            self.unet.eval()
        # Small sequential batches should not overwrite v11 normalization statistics.
        for m in self.unet.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x,state=None,ablate=False):
        u=self.unet;x1=u.inc(x);x2=u.down1(x1);x3=u.down2(x2);x4=u.down3(x3)
        if ablate:h8=h16=None
        else:
            h8,h16=(None,None) if state is None else state
            x4,h8=self.gru8(x4,h8)
        x5=u.down4(x4)
        if not ablate:x5,h16=self.gru16(x5,h16)
        z=u.up4(u.up3(u.up2(u.up1(x5,x4),x3),x2),x1);logits=u.outc(z)
        # Forecast uses only current causal features; next frame never enters this path.
        low=F.avg_pool2d(torch.cat((z,logits.softmax(1)),1),4)
        parameters=F.interpolate(self.transition(low),size=x.shape[-2:],mode='bilinear',align_corners=False)
        flow=self.max_flow*torch.tanh(parameters[:,:2]);delta=.75*torch.tanh(parameters[:,2:])
        h,w=x.shape[-2:];yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
        grid=torch.stack((xx,yy),-1)[None].expand(x.shape[0],-1,-1,-1).float()
        offset=flow.float().permute(0,2,3,1)*flow.new_tensor([2/(w-1),2/(h-1)]).float()
        probability=F.grid_sample(logits.float().softmax(1),grid+offset,mode='bilinear',padding_mode='border',align_corners=True)
        future=probability.clamp_min(1e-6).log()+delta.float()
        return dict(logits=logits,future=future,state=(h8,h16),flow=flow)
