"""Full original video with explicit invalid-frame gaps and causal target alignment."""
import argparse,csv,json
from pathlib import Path
import cv2,h5py,numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.pipeline.dataPrepare import fit
from models.unet.compare import overlay
from results.common import verify_video

def render(root,out):
    out.mkdir(parents=True,exist_ok=False);seq='subject21_supine_prepped_force_ultrasound';data=Path('data/datasets/PMC9883282')
    y0,y1,x0,x1=json.loads((data/'annotation_selection.json').read_text())['crop']
    with h5py.File(data/(seq+'.mat')) as f:raw=np.asarray(f['ultrasound_images'][:,x0:x1,y0:y1]).transpose(0,2,1)
    index=json.loads(Path('/tmp/uceeqz4_stage1_20260923/index.json').read_text());lookup={}
    for label,folder in [('v11','evaluation/baseline_joint'),('stage1','evaluation/candidate_joint')]:
        for p in (root/folder).glob(seq+'*.npz'):
            with np.load(p) as z:
                ids=z['ids'];current=z['current'];future=z['forecast'] if label=='stage1' else None
                for j,i in enumerate(ids):
                    frame=index['frames'][int(i)]['frame'];lookup.setdefault(frame,{})[label]=current[j]
                    if label=='stage1' and j+1<len(ids):lookup.setdefault(frame+1,{})['forecast']=future[j]
    rows=list(csv.DictReader((data/'meta_PMC9883282_1.csv').open()));labels={int(r['frame_index']):r for r in rows if r['sequence_id']==seq and r['frame_valid']=='true'}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);w,h,top,head=544,576,66,30;fps=8.;size=(w*2,top+(h+head)*2)
    video=cv2.VideoWriter(str(out/'comparison.webm'),cv2.VideoWriter_fourcc(*'VP80'),fps,size);assert video.isOpened()
    area=[];gt=[]
    for j,im in enumerate(raw):
        image,_,_=fit(im,None,(h,w),'letterbox');zero=np.zeros((h,w),dtype='uint8');m=lookup.get(j,{});valid='v11' in m and 'stage1' in m
        gray=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB);g=labels.get(j)
        if g:
            mask=np.array(Image.open(g['mask_path']));_,mask,_=fit(im,mask,(h,w),'letterbox');gids=np.zeros_like(mask);gids[mask==255]=1;gids[mask==128]=2
            for c in [1,2]:
                contours,_=cv2.findContours((gids==c).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(gray,contours,-1,(255,255,255),1)
                gt.append(dict(frame=j,class_id=c,area=int((gids==c).sum())))
        titles=['Observed image / reviewed GT','Frozen v11 | same joint output','Stage1 | current tracking',f'Forecast made at {j-1}, target {j}' if j and valid else 'No valid causal forecast']
        images=[gray,overlay(image,m.get('v11',zero)),overlay(image,m.get('stage1',zero)),overlay(image,m.get('forecast',zero))]
        canvas=Image.new('RGB',size,'#111827');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f'Dual-scale ConvGRU U-Net | subject21 supine | frame {j}/{len(raw)-1}',font=font,fill='white')
        draw.text((12,33),'Research candidate | blue vein, red artery, white reviewed GT | 8fps playback',font=font,fill='white')
        for k,(title,a) in enumerate(zip(titles,images)):
            xx=k%2*w;yy=top+k//2*(h+head);canvas.paste(Image.fromarray(a),(xx,yy+head));draw.text((xx+8,yy+5),title,font=font,fill='white')
            if not valid and k>0:draw.text((xx+12,yy+80),'EXCLUDED: no probe contact / state reset',font=font,fill='#ffb060')
        video.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
        if j in [118,338,543]:canvas.save(out/f'frame_{j:04d}.jpg')
        area.append(dict(frame=j,valid=valid,**{f'{name}_{c}':int((m[name]==c).sum()) if name in m and valid else None for name in ['v11','stage1','forecast'] for c in [1,2]}))
    video.release();decoded=verify_video(out/'comparison.webm',len(raw),size)
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
    for c,ax in enumerate(axes,1):
        for name,color in [('v11','#888888'),('stage1','#147bbe'),('forecast','#d6740c')]:ax.plot([a['frame'] for a in area],[a[f'{name}_{c}'] if a[f'{name}_{c}'] is not None else np.nan for a in area],label=name,color=color,lw=1)
        gs=[r for r in gt if r['class_id']==c];ax.scatter([r['frame'] for r in gs],[r['area'] for r in gs],s=16,c='black',label='reviewed GT');ax.set_ylabel(('Vein' if c==1 else 'Artery')+' area (pixels)');ax.legend(ncol=4);ax.axvspan(671,len(raw)-1,color='gray',alpha=.15)
    axes[-1].set_xlabel('Original frame index; grey = excluded no-contact interval');fig.tight_layout();fig.savefig(out/'area_traces.png',dpi=140);plt.close(fig)
    with (out/'area_traces.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(area[0]));writer.writeheader();writer.writerows(area)
    choice=json.loads((root/'dense_pre_test_selection.json').read_text());manifest=dict(sequence=seq,frames=len(raw),decoded_frames=decoded,fps=fps,valid_frames=sum(a['valid'] for a in area),checkpoint_sha256=choice['sha256'],epoch=choice['epoch'],alignment='forecast panel at frame j is prediction made at j-1; invalid targets and sequence starts have no forecast',selection=choice)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage1 双尺度 ConvGRU：完整视频</title><style>body{max-width:1200px;margin:24px auto;padding:0 18px;background:#111827;color:#eee;font:17px/1.7 sans-serif}video,img{width:100%}a{color:#93c5fd}button,select{font:inherit}</style><h1>Stage1：血管时序跟踪与预测</h1><p><strong>研究候选，未通过验收；当前默认仍为原 v11。</strong> <a href="../report/index.html">训练、对照与问题分析</a></p><p>受试者 21 仰卧视频，原标注序号 121–150。完整 756 帧，其中 671 帧有效；灰色区间为无接触帧，已排除并重置状态。新补标来自训练受试者 20，未将这个测试视频加入训练。</p><p>左上原图/人工白色轮廓，右上 v11，左下 Stage1 当前分割，右下上一帧作出的当前目标预测。所有模型使用相同 joint 输出处理，原始网络指标另列。视频是预先生成的，不会随标注编辑自动更新。</p><video id="v" controls preload="metadata" src="comparison.webm" poster="frame_0543.jpg"></video><p><button onclick="v.pause();v.currentTime=Math.max(0,v.currentTime-1/8)">上一帧</button> <button onclick="v.pause();v.currentTime=Math.min(v.duration,v.currentTime+1/8)">下一帧</button> <select aria-label="播放速度" onchange="v.playbackRate=Number(this.value)"><option value=".25">0.25倍</option><option value=".5">0.5倍</option><option selected value="1">1倍（8fps）</option></select></p><h2>面积变化与人工标签</h2><img src="area_traces.png" alt="各输出的动静脉面积曲线"><p><a href="area_traces.csv">逐帧面积</a> · <a href="manifest.json">模型来源、帧数及预测对齐</a></p></html>''')
    print('VIDEO COMPLETE',len(raw),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('results/temporal/runs/convgru_20260923'));a.add_argument('--output',type=Path,default=Path('results/temporal/generated/video'));args=a.parse_args();render(args.root,args.output)
