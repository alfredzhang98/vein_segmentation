"""Conservative no-contact exclusion for the local PMC ROI; preserves source files."""
import argparse,csv,hashlib,io,json,os,shutil
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
if __package__:
    from .frame_quality import reviewed_validity
else:
    from frame_quality import reviewed_validity

def tissue_stats(image):
    h,w=image.shape
    roi=image[int(h*.2):int(h*.9),int(w*.1):int(w*.9)]
    return float(roi.mean()),float((roi>20).mean())

def no_contact(image):
    mean,bright=tissue_stats(image)
    return mean<.5 and bright<.01

def segments(valid):
    out=[];start=None
    for i,v in enumerate([*valid,False]):
        if v and start is None:start=i
        elif not v and start is not None:out.append([start,i]);start=None
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    root=Path('data/datasets/PMC9883282');output=Path('data/audits/20260923_contact_cleanup');output.mkdir(parents=True,exist_ok=True)
    meta=root/'meta_PMC9883282_1.csv';raw=meta.read_bytes();reader=csv.DictReader(io.StringIO(raw.decode()));fields=reader.fieldnames;rows=list(reader)
    y0,y1,x0,x1=json.loads((root/'annotation_selection.json').read_text())['crop'];sequences={}
    for mat in sorted(root.glob('*.mat')):
        with h5py.File(mat) as f:
            valid=[];stats=[]
            images=np.asarray(f['ultrasound_images'][:,x0:x1,y0:y1]).transpose(0,2,1)
            for image in images:
                mean,bright=tissue_stats(image);valid.append(not (mean<.5 and bright<.01));stats.append([mean,bright])
        sequences[mat.stem]=dict(valid=valid,segments=segments(valid),stats=stats)
        print(mat.stem,'invalid',valid.count(False),'/',len(valid),flush=True)
    flags=reviewed_validity(sequences,rows)
    for seq,valid in flags.items():
        sequences[seq]['valid']=valid;sequences[seq]['segments']=segments(valid)
    invalid=[];kept_empty=[]
    for i,r in enumerate(rows,1):
        bad=not sequences[r['sequence_id']]['valid'][int(r['frame_index'])]
        mask=np.array(Image.open(r['mask_path']))
        if bad and mask.any() and r.get('quality_review')!='invalid':raise ValueError(f'Foreground labelled in suspected invalid frame {i}; inspect manually')
        r['frame_valid']='false' if bad else 'true'
        r['exclusion_reason']=(r.get('exclusion_reason') or 'no_contact_near_empty_tissue_roi') if bad else ''
        if bad:r['mask_status']='pass';invalid.append(i)
        elif not mask.any():kept_empty.append(i)
    payload=dict(rule='PMC cropped ROI: y20:90%, x10:90%; mean<0.5 AND fraction intensity>20 <0.01; visually reviewed labelled examples',sequences=sequences,excluded_annotation_numbers=invalid,retained_empty_mask_numbers=kept_empty,source_metadata_sha256=hashlib.sha256(raw).hexdigest())
    (output/'contact_audit.json').write_text(json.dumps(payload,indent=2))
    if a.apply:
        backup=Path('/home/uceeqz4/Project/ultrasound/temporal_archive/20260923_before_pipeline');backup.mkdir(parents=True,exist_ok=True)
        target=backup/meta.name
        if target.exists():raise FileExistsError('Refuse overwrite pre-cleaning backup')
        target.write_bytes(raw);shutil.copytree(root/'masks',backup/'pmc_masks');shutil.copytree(root/'images',backup/'pmc_images')
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields+[k for k in ['frame_valid','exclusion_reason'] if k not in fields]);writer.writeheader();writer.writerows(rows)
        assert meta.read_bytes()==raw,'Annotations changed during audit; rerun against latest snapshot'
        tmp=meta.with_suffix('.quality.tmp');tmp.write_text(stream.getvalue());os.replace(tmp,meta)
        (root/'frame_validity.json').write_text(json.dumps(payload,indent=2))
    print('EXCLUDED',len(invalid),invalid,'RETAINED EMPTY',len(kept_empty),kept_empty,flush=True)
if __name__=='__main__':main()
