"""Streaming causal evaluation; next-frame outputs are scored only after target arrives."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import cv2
from models.temporal.model import TrackingUNet
from models.temporal.data import Streams
from models.unet.postprocess import clean_labels

def mask_geometry(m):
    result=[]
    for c in [1,2]:
        yy,xx=np.where(m==c);result.append((len(xx),np.array([xx.mean(),yy.mean()]) if len(xx) else None))
    return result

def lag_diagnostics(traces):
    """Compare shifts on the same dense-GT support. Positive lag means delayed output."""
    grouped={}
    for r in traces:grouped.setdefault((r['dataset'],r['sequence'],r['class_id']),{})[r['frame']]=r
    rows=[]
    for (domain,sequence,c),frames in grouped.items():
        eligible=[]
        for t,r in frames.items():
            neighbors=[frames.get(t+k) for k in range(-3,4)]
            if all(n is not None and n['gt_area'] is not None and n['gt_area']>0 and n['area']>0 for n in neighbors):eligible.append(t)
        if len(eligible)<8:
            rows.append(dict(dataset=domain,sequence=sequence,class_id=c,n=len(eligible),estimable=False));continue
        errors=[]
        for lag in range(-3,4):
            e=[]
            for t in eligible:
                pred=frames[t+lag];gt=frames[t]
                e.append(((pred['cx']-gt['gt_cx'])/544)**2+((pred['cy']-gt['gt_cy'])/576)**2+.01*np.log(pred['area']/gt['gt_area'])**2)
            errors.append(float(np.mean(e)))
        rows.append(dict(dataset=domain,sequence=sequence,class_id=c,n=len(eligible),estimable=True,best_lag_frames=int(np.argmin(errors))-3,errors=errors))
    return rows

def motion_diagnostics(traces):
    """Error in labelled motion, rather than rewarding zero output change."""
    grouped={};totals={}
    for r in traces:grouped.setdefault((r['dataset'],r['sequence'],r['class_id']),{})[r['frame']]=r
    for (domain,sequence,c),frames in grouped.items():
        acc=totals.setdefault((domain,c),dict(labelled_pairs=0,missing_prediction_pairs=0,errors=[],area_errors=[]))
        for t,r in frames.items():
            before=frames.get(t-1)
            if before is None or not r['gt_area'] or not before['gt_area']:continue
            acc['labelled_pairs']+=1
            if not r['area'] or not before['area']:acc['missing_prediction_pairs']+=1;continue
            delta=np.array([r['cx']-before['cx'],r['cy']-before['cy']])
            target=np.array([r['gt_cx']-before['gt_cx'],r['gt_cy']-before['gt_cy']])
            acc['errors'].append(float(np.linalg.norm(delta-target)))
            acc['area_errors'].append(float(abs(np.log(r['area']/before['area'])-np.log(r['gt_area']/before['gt_area']))))
    return [dict(dataset=d,class_id=c,labelled_pairs=r['labelled_pairs'],missing_prediction_pairs=r['missing_prediction_pairs'],motion_error_px=float(np.mean(r['errors'])) if r['errors'] else None,log_area_change_error=float(np.mean(r['area_errors'])) if r['area_errors'] else None) for (d,c),r in totals.items()]

def evaluate(model,data,split,baseline=False,save=None,flow_baseline=False,ablate=False,joint=False):
    model.eval();metrics={};continuity={};saved={};traces=[]
    def update(domain,name,p,y):
        for c in [1,2]:
            key=(domain,name,c);r=metrics.setdefault(key,dict(n=0,dice_sum=0.,present_n=0,present_dice_sum=0.,absent_n=0,false_positive_frames=0,confused_pixels=0,gt_pixels=0,centroid_error_sum=0.,both_present_n=0,relative_area_error_sum=0.))
            pp=p==c;gg=y==c;a=int(pp.sum());b=int(gg.sum());tp=int((pp&gg).sum());dice=(2*tp+1e-6)/(a+b+1e-6)
            r['n']+=1;r['dice_sum']+=dice;r['gt_pixels']+=b;r['confused_pixels']+=int(((p==(3-c))&gg).sum())
            if b:
                r['present_n']+=1;r['present_dice_sum']+=dice;r['relative_area_error_sum']+=abs(a-b)/b
                if a:
                    py,px=np.where(pp);gy,gx=np.where(gg);r['centroid_error_sum']+=float(np.hypot(px.mean()-gx.mean(),py.mean()-gy.mean()));r['both_present_n']+=1
            else:r['absent_n']+=1;r['false_positive_frames']+=int(a>0)
    with torch.inference_mode():
        for s in data.sequences:
            if s['split']!=split:continue
            state=None;previous_pred=None;previous_forecast=None;previous_flow_pred=None;previous_image=None;last_geometry={};seqm=[];seqf=[]
            for i in s['ids']:
                row=data.frames[i];raw=np.array(data.images[i]);x=torch.from_numpy(raw).to(data.device).float()[None,None]/127.5-1
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(data.device).startswith('cuda')):out=model(x,None if ablate else state,ablate=baseline)
                state=out['state'];prob=out['logits'].float().softmax(1)[0].cpu().numpy();p=prob.argmax(0).astype('uint8');future=out['future'].argmax(1)[0].cpu().numpy().astype('uint8') if not baseline else p.copy()
                if joint:p=clean_labels(p);future=clean_labels(future)
                if data.labelled[i]:
                    y=np.asarray(data.masks[i]);update(s['dataset'],'current',p,y)
                    if previous_pred is not None:update(s['dataset'],'persistence',previous_pred,y)
                    if previous_forecast is not None:update(s['dataset'],'forecast',previous_forecast,y)
                    if previous_flow_pred is not None:update(s['dataset'],'flow',previous_flow_pred,y)
                for name,mask in [('current',p),('forecast',future)]:
                    if name=='forecast' and i==s['ids'][-1]:continue
                    gs=mask_geometry(mask)
                    for c,(area,center) in enumerate(gs,1):
                        key=(s['dataset'],name,c);r=continuity.setdefault(key,dict(pairs=0,both_present=0,jumps20=0,area_jumps50=0,appears=0,disappears=0))
                        old=last_geometry.get((name,c))
                        if old is not None:
                            a0,c0=old;r['pairs']+=1;r['appears']+=int(not a0 and area>0);r['disappears']+=int(a0>0 and not area)
                            if area and a0:r['both_present']+=1;r['jumps20']+=int(np.linalg.norm(center-c0)>20);r['area_jumps50']+=int(max(area/a0,a0/area)>1.5)
                        last_geometry[name,c]=(area,center)
                        if name=='current':
                            gtarea,gtcenter=mask_geometry(np.asarray(data.masks[i]))[c-1] if data.labelled[i] else (None,None)
                            traces.append(dict(dataset=s['dataset'],sequence=s['sequence'],frame=row['frame'],class_id=c,area=area,cx=float(center[0]) if area else None,cy=float(center[1]) if area else None,gt_area=gtarea,gt_cx=float(gtcenter[0]) if gtcenter is not None else None,gt_cy=float(gtcenter[1]) if gtcenter is not None else None))
                if flow_baseline:
                    if previous_image is None:previous_flow_pred=p.copy()
                    else:
                        small=cv2.resize(raw,(136,144));before=cv2.resize(previous_image,(136,144))
                        f=cv2.calcOpticalFlowFarneback(small,before,None,.5,3,15,3,5,1.2,0)
                        f=cv2.resize(f,(544,576))*4;yy,xx=np.mgrid[:576,:544].astype('float32')
                        transported=cv2.remap(prob.transpose(1,2,0),xx+f[...,0],yy+f[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        previous_flow_pred=transported.argmax(-1).astype('uint8')
                        if joint:previous_flow_pred=clean_labels(previous_flow_pred)
                if save and s['dataset']=='pmc9883282':seqm.append(p);seqf.append(future)
                previous_pred=p;previous_forecast=future;previous_image=raw
            if save and seqm:saved[s['sequence']+'_'+str(data.frames[s['ids'][0]]['frame'])]=dict(ids=np.array(s['ids']),current=np.stack(seqm),forecast=np.stack(seqf))
    rows=[]
    for (d,m,c),r in metrics.items():
        r.update(dataset=d,model=m,class_id=c,dice=r['dice_sum']/r['n'],present_dice=r['present_dice_sum']/max(1,r['present_n']),centroid_error=r['centroid_error_sum']/max(1,r['both_present_n']),relative_area_error=r['relative_area_error_sum']/max(1,r['present_n']),class_confusion=r['confused_pixels']/max(1,r['gt_pixels']));rows.append(r)
    result=dict(split=split,summary=rows,continuity=[dict(dataset=d,model=m,class_id=c,**r) for (d,m,c),r in continuity.items()],traces=traces,lag=lag_diagnostics(traces),motion=motion_diagnostics(traces))
    if save:
        save=Path(save);save.mkdir(parents=True,exist_ok=True)
        for name,p in saved.items():np.savez_compressed(save/(name+'.npz'),**p)
        (save/'summary.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--cache',default='/tmp/uceeqz4_stage1_20260923');ap.add_argument('--split',default='test');ap.add_argument('--output',required=True);ap.add_argument('--baseline',action='store_true');ap.add_argument('--ablate',action='store_true');ap.add_argument('--joint',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2);ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.baseline:model=TrackingUNet(ck['config']);model.unet.load_state_dict(ck['model'])
    else:model=TrackingUNet(**ck['config']);model.load_state_dict(ck['model'])
    data=Streams(args.cache,'cuda:0');model.to(data.device)
    index_sha=hashlib.sha256((Path(args.cache)/'index.json').read_bytes()).hexdigest()
    if not args.baseline and ck.get('index_sha256')!=index_sha:raise ValueError('Checkpoint and sequence/label index differ')
    result=evaluate(model,data,args.split,baseline=args.baseline,ablate=args.ablate,save=args.output,flow_baseline=True,joint=args.joint)
    result.update(checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),epoch=ck['epoch'],index_sha256=index_sha,ablation=args.ablate,baseline=args.baseline,postprocess='joint' if args.joint else 'raw')
    (Path(args.output)/'summary.json').write_text(json.dumps(result,indent=2))
    print([(r['dataset'],r['model'],r['class_id'],round(r['dice'],4)) for r in result['summary']],flush=True)
if __name__=='__main__':main()
