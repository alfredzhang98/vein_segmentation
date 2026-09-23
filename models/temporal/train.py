"""Eight-frame causal learning with per-class geometry and labelled temporal differences."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from models.temporal.model import TrackingUNet
from models.temporal.data import Streams
from models.temporal.losses import segmentation,geometry_error,change_error
from models.temporal.evaluate import evaluate

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selection(result,baseline):
    rows={(r['dataset'],r['model'],r['class_id']):r for r in result['summary']};base={(r['dataset'],r['model'],r['class_id']):r for r in baseline['summary']}
    failures=[];curr=[];future=[]
    for d in ['musv','pmc9883282']:
        for c in [1,2]:
            a=rows[d,'current',c];b=base[d,'current',c];f=rows[d,'forecast',c];bp=base[d,'persistence',c]
            curr.append(a['dice']);future.append(f['dice'])
            if a['dice']<b['dice']-.005:failures.append(f'{d}/{c}/current_dice')
            if a['false_positive_frames']>b['false_positive_frames']+1:failures.append(f'{d}/{c}/false_positives')
            if f['dice']<bp['dice']-.005:failures.append(f'{d}/{c}/future_vs_persistence')
    rates=[(r['jumps20']+r['area_jumps50'])/max(r['both_present'],1) for r in result['continuity'] if r['model']=='current']
    return float(np.mean(curr)+.5*np.mean(future)-.03*np.mean(rates)),failures

def train(cfg):
    data=Streams(cfg['cache'],'cuda:0');data.require_current_annotations()
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=False);(run/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    windows=data.windows(8,require_dense=cfg.get('dense_clips_only',False));assert all(windows.values())
    (run/'sampling.json').write_text(json.dumps(dict(windows={d:len(v) for d,v in windows.items()},dense_clips_only=cfg.get('dense_clips_only',False)),indent=2))
    ck=torch.load(cfg['init_checkpoint'],map_location='cpu',weights_only=False);model=TrackingUNet(ck['config'],hidden=cfg['hidden']).to(data.device);model.unet.load_state_dict(ck['model'])
    if cfg.get('freeze_unet'):
        for parameter in model.unet.parameters():parameter.requires_grad_(False)
    backbone_sha=digest(cfg['init_checkpoint']);index_sha=digest(Path(cfg['cache'])/'index.json')
    opt=torch.optim.AdamW([dict(params=model.unet.parameters(),lr=cfg['backbone_lr']),dict(params=[p for n,p in model.named_parameters() if not n.startswith('unet.')],lr=cfg['temporal_lr'])],weight_decay=.001)
    baseline=evaluate(model,data,'val',baseline=True,flow_baseline=True);(run/'baseline_val.json').write_text(json.dumps(baseline,indent=2))
    best=-1e9;best_any=-1e9;stale=0;start=time.monotonic();selected=0
    for epoch in range(cfg['epochs']+1):
        loss_total=0;grad_max=0
        if epoch:
            model.train()
            for step in range(cfg['clips_per_epoch']//cfg['batch_size']):
                batches=[]
                for b in range(cfg['batch_size']):
                    domain='pmc9883282' if random.random()<.4 else 'musv';ids=random.choice(windows[domain]);batches.append(data.clip(ids,augment=True))
                x=torch.stack([b[0] for b in batches]);y=torch.stack([b[1] for b in batches]);valid=torch.stack([b[2] for b in batches]);state=None;previous=None
                terms={k:[] for k in ['current','future','geometry','change']};opt.zero_grad(set_to_none=True)
                for t in range(8):
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x[:,t],state)
                    state=out['state'];v=valid[:,t];nv=valid[:,t+1]
                    if v.any():
                        terms['current'].append(segmentation(out['logits'][v],y[v,t]));terms['geometry'].append(geometry_error(out['logits'][v],y[v,t]))
                    if nv.any():
                        terms['future'].append(segmentation(out['future'][nv],y[nv,t+1]));terms['geometry'].append(geometry_error(out['future'][nv],y[nv,t+1]))
                    if previous is not None:
                        paired=valid[:,t-1]&v
                        if paired.any():terms['change'].append(change_error(previous[paired],out['logits'][paired],y[paired,t-1],y[paired,t]))
                    previous=out['logits']
                loss=sum(weight*torch.stack(terms[k]).mean() for k,weight in [('current',1.),('future',.7),('geometry',.15),('change',.2)] if terms[k])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();loss_total+=float(loss.detach());grad_max=max(grad_max,float(norm))
                if (step+1)%30==0:print('STEP',epoch,step+1,'loss',loss_total/(step+1),flush=True)
        result=evaluate(model,data,'val');score,failures=selection(result,baseline)
        (run/f'val_epoch{epoch:02d}.json').write_text(json.dumps(result,indent=2))
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=model.config,epoch=epoch,score=score,backbone_sha256=backbone_sha,index_sha256=index_sha,training_config=cfg)
        # Epoch zero is a reference, never a trained diagnostic candidate.
        if epoch and score>best_any:best_any=score;torch.save(payload,run/'best_unconstrained.pth')
        if epoch:torch.save(payload,run/'last.pth')
        if not failures and score>best+.0003:
            best=score;stale=0;selected=epoch;torch.save(payload,run/'selected.pth')
        elif epoch:stale+=1
        record=dict(epoch=epoch,score=score,eligible=not failures,failures=failures,loss=loss_total/max(1,cfg['clips_per_epoch']//cfg['batch_size']) if epoch else None,gradient_norm_max=grad_max,seconds=time.monotonic()-start)
        with (run/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
    (run/'complete.json').write_text(json.dumps(dict(selected_epoch=selected,best=best,elapsed=time.monotonic()-start),indent=2));print('COMPLETE',selected,best,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);train(json.loads(Path(ap.parse_args().config).read_text()))
