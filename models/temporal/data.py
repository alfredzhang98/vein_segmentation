import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

class Streams:
    def __init__(self,path,device='cpu'):
        self.path=Path(path);self.index=json.loads((self.path/'index.json').read_text());self.frames=self.index['frames'];self.sequences=self.index['sequences'];self.device=device
        self.images=np.load(self.path/'images.npy',mmap_mode='r');self.masks=np.load(self.path/'masks.npy',mmap_mode='r')
        self.labelled=np.array([r['labelled'] for r in self.frames],dtype=bool)
    def require_current_annotations(self, metadata=Path('data/datasets/PMC9883282/meta_PMC9883282_1.csv')):
        """Training must not silently reuse arrays from before a review/exclusion."""
        sources = {str(metadata): self.index.get('metadata_sha256'),
                   str(Path(metadata).parent/'frame_validity.json'): self.index.get('contact_sha256'),
                   **self.index.get('label_sha256', {})}
        for source, expected in sources.items():
            path=Path(source)
            if not expected or not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Annotations or quality decisions changed; rebuild the temporal cache with models.temporal.prepare before training: '+source)
    def windows(self,length=8,require_dense=False):
        groups={'musv':[],'pmc9883282':[]}
        for s in self.sequences:
            if s['split']!='train':continue
            ids=s['ids']
            for start in range(len(ids)-length):
                window=ids[start:start+length+1]
                labels=self.labelled[window]
                if (labels.all() if require_dense else labels.any()):groups[s['dataset']].append(window)
        return groups
    def clip(self,ids,augment=False):
        x=torch.from_numpy(np.array(self.images[ids])).to(self.device).float()[:,None]/255
        y=torch.from_numpy(np.array(self.masks[ids])).to(self.device).long()
        if augment:
            # One rigid/uniform-scale transform for the whole clip, including future labels.
            angle=(torch.rand((),device=self.device)-.5)*.35;scale=.9+.2*torch.rand((),device=self.device)
            sign=torch.where(torch.rand((),device=self.device)<.5,-1.,1.)
            theta=torch.zeros((1,2,3),device=self.device)
            theta[0,0,0]=sign*torch.cos(angle)*scale;theta[0,0,1]=-torch.sin(angle)*scale
            theta[0,1,0]=sign*torch.sin(angle)*scale;theta[0,1,1]=torch.cos(angle)*scale
            theta[0,:,2]=(torch.rand(2,device=self.device)-.5)*.06
            grid=F.affine_grid(theta.expand(len(ids),-1,-1),x.shape,align_corners=False)
            x=F.grid_sample(x,grid,align_corners=False,padding_mode='zeros')
            y=F.grid_sample(y[:,None].float(),grid,mode='nearest',align_corners=False)[:,0].long()
            gamma=.8+.4*torch.rand((),device=self.device);gain=.9+.2*torch.rand((),device=self.device)
            x=(gain*x.clamp_min(0).pow(gamma)+(torch.rand((),device=self.device)-.5)*.08).clamp(0,1)
            if torch.rand(())<.4:x=(x*(1+.06*torch.randn_like(x))).clamp(0,1)
        return x*2-1,y,torch.tensor(self.labelled[ids],device=self.device)
