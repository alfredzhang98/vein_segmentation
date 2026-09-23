"""Render the longest local Mus-V test sequence without selecting by model score."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from gpu_utils import pick_idle_gpu
from types import SimpleNamespace
from models.unet.model import UNet
from models.unet.compare import overlay, cleaned, frame_metrics, sha256
from results.common import verify_video


def main():
    pick_idle_gpu()
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache = Path('/tmp/uceeqz4_stage1_20260923')
    data = SimpleNamespace(frames=json.loads((cache/'index.json').read_text())['frames'],images=np.load(cache/'images.npy',mmap_mode='r'),masks=np.load(cache/'masks.npy',mmap_mode='r'))
    groups = {}
    for i, row in enumerate(data.frames):
        if row['dataset'] == 'musv' and row['split'] == 'test':
            groups.setdefault(row['sequence'], []).append(i)
    sequence, ids = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[0]
    ids.sort(key=lambda i: data.frames[i]['frame'])
    assert all(data.frames[b]['frame'] == data.frames[a]['frame'] + 1 for a, b in zip(ids, ids[1:]))
    out = Path('results/unet/generated/musv_review')
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads(Path('results/unet/runs/v11_refresh_20260923/checkpoint_reference.json').read_text())
    candidate = reference['archived']
    paths = {'original_v11': 'models/unet/checkpoints/unet_v11.pth', 'new_v11_candidate': candidate}
    predictions, provenance = {}, {}
    for name, path in paths.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ck['config']
        model = UNet(1, 3, bilinear=cfg.get('bilinear', False), base_ch=cfg.get('base_ch', 64), dropout=cfg.get('dropout', 0)).to(device).eval()
        model.load_state_dict(ck['model'])
        pred = []
        with torch.inference_mode():
            for start in range(0, len(ids), 8):
                x = torch.from_numpy(np.array(data.images[ids[start:start+8]])).to(device).float()[:,None]/127.5-1
                with torch.autocast(device.type, enabled=device.type == 'cuda'):
                    logits = model(x)
                pred.extend(cleaned(p, 'full3', 'joint') for p in logits.argmax(1).cpu().numpy().astype('uint8'))
        predictions[name] = np.stack(pred)
        provenance[name] = dict(path=path, sha256=sha256(path), epoch=ck['epoch'])
        del model, ck
    w, h, title, header = 544, 576, 30, 64
    size = (2*w, header+2*(h+title))
    fps = 8.
    writer = cv2.VideoWriter(str(out/'comparison.webm'), cv2.VideoWriter_fourcc(*'VP80'), fps, size)
    raw_size = Image.open(data.frames[ids[0]]['source']).size
    raw_writer = cv2.VideoWriter(str(out/'original.webm'), cv2.VideoWriter_fourcc(*'VP80'), fps, raw_size)
    assert writer.isOpened() and raw_writer.isOpened()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    rows = []
    for j, idx in enumerate(ids):
        record = data.frames[idx]
        native = np.array(Image.open(record['source']).convert('RGB'))
        assert native.shape[1::-1] == raw_size
        raw_writer.write(cv2.cvtColor(native, cv2.COLOR_RGB2BGR))
        gray = np.array(data.images[idx]); gt = np.array(data.masks[idx])
        canvas = Image.new('RGB', size, '#111827'); draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), f'Mus-V test | {sequence} | source frame {record["frame"]} / {len(ids)}', font=font, fill='white')
        draw.text((12, 32), 'Blue: vein | Red: artery | 8fps playback, acquisition timing unknown', font=font, fill='white')
        panels = [('Image', np.repeat(gray[..., None], 3, -1)), ('Dataset annotation (unmodified)', overlay(gray, gt))]
        for name, title_text in [('original_v11','Original v11'), ('new_v11_candidate','New v11 candidate (not promoted)')]:
            p = predictions[name][j]
            panels.append((title_text, overlay(gray, p)))
            for metric in frame_metrics(p, gt, 'full3'):
                rows.append(dict(frame=record['frame'], model=name, **metric))
        for k, (label, arr) in enumerate(panels):
            xx, yy = (k % 2)*w, header+(k//2)*(h+title)
            draw.text((xx+8, yy+5), label, font=font, fill='white')
            canvas.paste(Image.fromarray(arr), (xx, yy+title))
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
        if j in [0, len(ids)//2, len(ids)-1]:
            canvas.save(out/f'frame_{record["frame"]:04d}.jpg')
    writer.release(); raw_writer.release()
    decoded = verify_video(out/'comparison.webm', len(ids), size)
    verify_video(out/'original.webm', len(ids), raw_size)
    with (out/'per_frame.csv').open('w', newline='') as f:
        writer_csv=csv.DictWriter(f, fieldnames=list(rows[0]));writer_csv.writeheader();writer_csv.writerows(rows)
    scores = {name: {cl: float(np.mean([r['dice'] for r in rows if r['model']==name and r['vessel_class']==cl])) for cl in ['vein','artery']} for name in paths}
    manifest = dict(sequence=sequence, split='test', selection='Longest sequence in local Mus-V test split, ties sorted by sequence ID; no score-based selection', frames=len(ids), decoded_frames=decoded, source_frame_start=data.frames[ids[0]]['frame'], source_frame_end=data.frames[ids[-1]]['frame'], playback_fps=fps, acquisition_fps=None, source_paths=[data.frames[i]['source'] for i in ids], preprocessing='comparison: original project 576x544 letterbox; pure video: original native images', postprocessing='same joint policy for both models', annotations='Original dataset annotations, unchanged', models=provenance, sequence_dice=scores)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    table=''.join(f'<tr><td>{label}</td><td>{scores[name]["vein"]:.4f}</td><td>{scores[name]["artery"]:.4f}</td></tr>' for name,label in [('original_v11','原 v11'),('new_v11_candidate','新 v11 候选')])
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mus-V 连续片段与标注检查</title><style>body{{max-width:1150px;margin:24px auto;padding:0 18px;background:#111827;color:#eee;font:17px/1.7 sans-serif}}video{{width:100%}}#raw{{max-height:650px}}a{{color:#93c5fd}}button,select{{font:inherit}}td,th{{padding:6px 20px;text-align:left}}</style><h1>Mus-V：原图、数据集标注与新旧 v11</h1><p>本地测试序列 <code>{sequence}</code>，连续 {len(ids)} 帧。按测试集最长片段选取，没有按模型分数挑选。数据源路径含 <code>Videos/valid</code>，本项目将该序列分在 test，来源详见清单。</p><p>默认 8fps 慢放，约 {len(ids)/fps:.0f} 秒；原采集时间间隔未校准，这是一段按原序号拼接的连续图像。不能用这一段判断整个数据集质量。</p><h2>四格对照</h2><p>左上原图，右上数据集原始标注，左下原 v11，右下新 v11 候选。蓝色静脉，红色动脉；两模型使用相同后处理，标注没有平滑或修补。</p><video id="v" controls loop preload="metadata" src="comparison.webm" poster="frame_0061.jpg"></video><p><button onclick="v.pause();v.currentTime=Math.max(0,v.currentTime-1/8)">上一帧</button> <button onclick="v.pause();v.currentTime=Math.min(v.duration,v.currentTime+1/8)">下一帧</button> <select aria-label="播放速度" onchange="v.playbackRate=Number(this.value)"><option value=".25">0.25 倍</option><option value=".5">0.5 倍</option><option value="1" selected>1 倍（8fps）</option></select></p><h2>纯原图</h2><video id="raw" controls loop preload="metadata" src="original.webm"></video><p>检查时重点看：原图组织边界是否可辨；标注是否漏掉可见血管；图像变化不大时标注轮廓是否突然改变。模型和标注不一致本身不能证明标注错误。</p><h2>仅本片段 Dice</h2><table><tr><th>模型</th><th>静脉</th><th>动脉</th></tr>{table}</table><p>这里的数值只描述这一段，不是整个测试集。当前默认仍为原 v11。</p><p><a href="original.webm" download>下载纯原图视频</a> · <a href="comparison.webm" download>下载对照视频</a> · <a href="per_frame.csv">逐帧指标</a> · <a href="manifest.json">来源与模型</a> · <a href="../../../temporal/generated/report/index.html">完整实验报告</a></p></html>''')
    print(json.dumps(dict(output=str(out),sequence=sequence,frames=decoded,scores=scores),indent=2),flush=True)

if __name__ == '__main__':
    main()
