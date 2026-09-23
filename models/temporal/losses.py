"""Label-grounded geometry and true temporal changes; absent is different from unlabelled."""
import torch
from torch.nn import functional as F
from models.losses import partial_label_loss

def geometry(p):
    p=p[:,1:3].float();h,w=p.shape[-2:];mass=p.sum((-2,-1)).clamp_min(1e-6)
    x=torch.linspace(0,1,w,device=p.device);y=torch.linspace(0,1,h,device=p.device)
    cx=(p.sum(-2)*x).sum(-1)/mass;cy=(p.sum(-1)*y).sum(-1)/mass
    return torch.stack((cx,cy),-1),mass/(h*w)

def target_geometry(y):
    return geometry(F.one_hot(y.long(),3).permute(0,3,1,2).float())

def geometry_error(logits,y):
    center,area=geometry(logits.float().softmax(1));gtc,gta=target_geometry(y)
    exists=gta>0
    ce=((center-gtc).abs().mean(-1)*exists).sum()/exists.sum().clamp_min(1)
    ae=F.smooth_l1_loss(area*100,gta*100)
    return ce+ae

def change_error(previous,current,yp,yc):
    pc,pa=geometry(previous.float().softmax(1));cc,ca=geometry(current.float().softmax(1))
    gpc,gpa=target_geometry(yp);gcc,gca=target_geometry(yc)
    visible=(gpa>0)&(gca>0)
    center=(((cc-pc)-(gcc-gpc)).abs().mean(-1)*visible).sum()/visible.sum().clamp_min(1)
    area=F.smooth_l1_loss((ca-pa)*100,(gca-gpa)*100)
    return center+area

def segmentation(logits,y):
    return partial_label_loss(logits,y[:,None].long(),['full3']*len(y))
