"""Rebuild full causal streams from real frames and current reviewed masks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from data.pipeline.dataPrepare import fit,ReadDataset,DataPipeline
from data.pipeline.frame_quality import reviewed_validity

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(out):
    if out.exists():raise FileExistsError(out)
    root=Path('data/datasets/PMC9883282');meta=root/'meta_PMC9883282_1.csv';validity=json.loads((root/'frame_validity.json').read_text())
    rows=list(csv.DictReader(meta.open()));valid_flags=reviewed_validity(validity['sequences'],rows)
    labels={(r['sequence_id'],int(r['frame_index'])):r for r in rows if r['mask_status'].lower() in ('true','test') and valid_flags[r['sequence_id']][int(r['frame_index'])]}
    old=dict(frames=[],sequences=[]);mus_rows=DataPipeline('musv')._load_rows_from_musv()
    for split in ['train','val','test']:
        groups={}
        for i,r in enumerate(mus_rows[split]):
            groups.setdefault(r['seq'],[]).append(dict(dataset='musv',sequence=r['seq'],split=split,frame=int(Path(r['filename']).stem.rsplit('_',1)[1]),label_index=i,label_mode='full3',source=r['relative_path']))
        for r in rows:
            if r['split']==split and r['sequence_id'] not in groups:
                seq=r['sequence_id'];groups[seq]=[dict(dataset='pmc9883282',sequence=seq,split=split,frame=i,label_index=-1,label_mode='full3',source=str(root/r['source_mat'])) for i in range(len(validity['sequences'][seq]['valid']))]
        for seq,part in groups.items():
            ids=list(range(len(old['frames']),len(old['frames'])+len(part)));old['frames'].extend(part)
            old['sequences'].append(dict(dataset=part[0]['dataset'],sequence=seq,split=split,ids=ids))
    source_sha=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    frames=[];sequences=[]
    for s in old['sequences']:
        if s['dataset'] not in ('musv','pmc9883282'):continue
        groups=[];group=[]
        for i in s['ids']:
            r=old['frames'][i]
            good=s['dataset']=='musv' or valid_flags[s['sequence']][r['frame']]
            if not good:
                if group:groups.append(group);group=[]
                continue
            group.append(r)
        if group:groups.append(group)
        for part in groups:
            ids=[]
            for r in part:
                ids.append(len(frames));nr=dict(r)
                if s['dataset']=='pmc9883282':nr['labelled']=(s['sequence'],r['frame']) in labels
                else:nr['labelled']=True
                frames.append(nr)
            sequences.append(dict(dataset=s['dataset'],sequence=s['sequence'],split=s['split'],ids=ids))
    out.mkdir(parents=True);n=len(frames);shape=(n,576,544)
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype='uint8',shape=shape)
    masks=np.lib.format.open_memmap(out/'masks.npy',mode='w+',dtype='uint8',shape=shape)
    mus={s:ReadDataset('validation' if s=='val' else s,1,'musv',augment=False) for s in ['train','val','test']}
    mask_hashes={};handles={};roi=json.loads((root/'annotation_selection.json').read_text())['crop'];y0,y1,x0,x1=roi
    for s in sequences:
        if s['dataset']=='pmc9883282':
            if s['sequence'] not in handles:
                with h5py.File(root/(s['sequence']+'.mat')) as f:handles={s['sequence']:np.asarray(f['ultrasound_images'][:,x0:x1,y0:y1]).transpose(0,2,1)}
            raw=handles[s['sequence']]
        for i in s['ids']:
            r=frames[i]
            if r['dataset']=='musv':
                ds=mus[r['split']];images[i]=ds.images[r['label_index']];masks[i]=ds.masks[r['label_index']]
            else:
                im=raw[r['frame']];mask=np.zeros_like(im)
                row=labels.get((r['sequence'],r['frame']))
                if row:
                    p=Path(row['mask_path']);m=np.array(Image.open(p));assert m.shape==im.shape and set(np.unique(m))<={0,128,255}
                    mask[m==255]=1;mask[m==128]=2;mask_hashes[str(p)]=digest(p)
                im,mask,_=fit(im,mask,(576,544),'letterbox');images[i]=im;masks[i]=mask
        print('cached',s['dataset'],s['sequence'],len(s['ids']),flush=True)
    images.flush();masks.flush()
    payload=dict(frames=frames,sequences=sequences,shape=[576,544],metadata_sha256=digest(meta),contact_sha256=digest(root/'frame_validity.json'),label_sha256=mask_hashes,source_index_sha256=source_sha,time_unit='one native source sample; no calibrated milliseconds')
    (out/'index.json').write_text(json.dumps(payload,indent=2));print('COMPLETE',len(frames),len(sequences),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);prepare(a.parse_args().output)
