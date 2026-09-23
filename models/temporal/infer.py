"""Streaming interface: one observed frame per call, forecast refers to the next sample."""
import cv2,numpy as np,torch
from models.temporal.model import TrackingUNet
from models.unet.infer import fit,unfit_mask
from data.pipeline.audit_contact import no_contact

class VesselTracker:
    def __init__(self,checkpoint,device=None):
        self.device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=TrackingUNet(**ck['config']).to(self.device).eval();self.model.load_state_dict(ck['model']);self.reset()
    def reset(self):self.state=None;self.shape=None
    @torch.inference_mode()
    def predict(self,image,dataset='pmc9883282'):
        if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if dataset=='pmc9883282' and no_contact(image):
            self.reset();return dict(valid=False,current=None,future=None,reason='no_contact')
        if self.shape is not None and image.shape!=self.shape:self.reset()
        self.shape=image.shape
        x,_,params=fit(image,None,(576,544),'letterbox');t=torch.from_numpy(x).to(self.device).float()[None,None]/127.5-1
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(self.device).startswith('cuda')):r=self.model(t,self.state)
        self.state=r['state']
        return dict(valid=True,current=unfit_mask(r['logits'].argmax(1)[0].cpu().numpy().astype('uint8'),params),future=unfit_mask(r['future'].argmax(1)[0].cpu().numpy().astype('uint8'),params))
