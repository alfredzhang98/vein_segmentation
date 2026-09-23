"""Paired checkpoint evaluation on identical cached splits, with a browsable report.

Run from vein_segmentation/. Does not train, rebuild caches or tune postprocessing.
Use --output results/unet/generated/<experiment> for local, Git-ignored artifacts.
Keep durable conclusions in results/unet/BASELINE_COMPARISON_CN.md.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpu_utils import pick_idle_gpu

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from data.pipeline.dataPrepare import ReadDataset, DataPipeline, check_stale
from models.unet.model import UNet
from models.unet.postprocess import clean_binary, clean_labels, complete_region


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def semantic_masks(labels, mode):
    if mode == 'full3':
        return {'vein': labels == 1, 'artery': labels == 2, 'vessel': labels > 0}
    if mode == 'artery':
        return {'artery': labels == 2}
    return {'vessel': labels > 0}


def cleaned(labels, mode, policy="largest"):
    if policy == "joint":
        if mode == "vessel":
            return complete_region(labels > 0).astype(np.uint8) * 3
        return clean_labels(labels)
    if mode == 'vessel':
        return clean_binary(labels > 0).astype(np.uint8) * 3
    out = np.zeros_like(labels)
    for cid in (1, 2):
        out[clean_binary(labels == cid)] = cid
    return out


def frame_metrics(pred, truth, mode):
    result = []
    for name, g in semantic_masks(truth, mode).items():
        p = semantic_masks(pred, mode)[name]
        ng, npred, tp = int(g.sum()), int(p.sum()), int((g & p).sum())
        result.append(dict(vessel_class=name, gt_pixels=ng, pred_pixels=npred, tp=tp,
                           dice=2 * tp / (ng + npred) if ng + npred else 1.0))
    return result


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r['dataset'], r['model'], r['processing'], r['vessel_class'])
        groups.setdefault(key, []).append(r)
    result = []
    for (dataset, model, processing, cls), items in groups.items():
        present = [r for r in items if r['gt_pixels'] > 0]
        absent = [r for r in items if r['gt_pixels'] == 0]
        result.append(dict(dataset=dataset, model=model, processing=processing, vessel_class=cls,
                           n=len(items), dice=float(np.mean([r['dice'] for r in items])),
                           present_n=len(present),
                           present_dice=float(np.mean([r['dice'] for r in present])) if present else None,
                           absent_n=len(absent), absent_fp_n=sum(r['pred_pixels'] > 0 for r in absent),
                           absent_mean_fp_pixels=float(np.mean([r['pred_pixels'] for r in absent])) if absent else None))
    return result


def overlay(gray, labels):
    palette = np.array([[0, 0, 0], [50, 130, 255], [245, 65, 65], [45, 205, 110]])
    image = np.repeat(gray[..., None], 3, axis=-1)
    return np.where((labels > 0)[..., None], .5 * image + .5 * palette[labels], image).astype('uint8')


def save_panel(path, gray, gt, preds, mode, names, policy="largest"):
    panels = [('Image', np.repeat(gray[..., None], 3, axis=-1)), ('Ground truth', overlay(gray, gt))]
    for name in names:
        raw = preds[name]
        display_raw = (raw > 0).astype('uint8') * 3 if mode == 'vessel' else raw
        panels += [(name + ' raw', overlay(gray, display_raw)),
                   (name + ' largest', overlay(gray, cleaned(raw, mode)))]
        if policy == 'joint':
            panels.append((name + ' repaired', overlay(gray, cleaned(raw, mode, policy))))
    w, h = 360, round(gray.shape[0] * 360 / gray.shape[1])
    canvas = Image.new('RGB', (w * len(panels), h + 32), '#111827')
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(panels):
        draw.text((i * w + 10, 10), title, fill='white')
        canvas.paste(Image.fromarray(arr).resize((w, h)), (i * w, 32))
    canvas.save(path)



def final_comparison_section(summary, names, policy):
    """Headline results always compare the same final policy for both models."""
    esc = html.escape
    steps = [
        '模型输出：argmax 得到背景、静脉、动脉三个互斥类别。',
        '找主体：分别找动、静脉面积至少4像素的最大8连通区域，作为类别锚点。',
        '修复局部错色：检查原始血管并集；某连通区域仅含一种类别锚点且该类占比≥60%时，把局部错色归回主体。两类主体相接时不强行合并。',
        '清理区域：重归类后，各类只保留面积至少4像素的最大区域，删除多余孤岛，允许为空。',
        '补形状：对保留区域做半径2的5×5椭圆核闭运算，再填封闭孔洞；不做开运算，空输入不补出血管。',
        '处理竞争：不覆盖另一类保留像素；新增区域重叠时按距补全前区域的距离归属，等距时归静脉。',
        '最终筛选与输出：再次保证每类最多一个区域；输出静脉、动脉及二者并集，血管并集不是第四个训练类别。',
    ]
    chunks = ['<section id="final-comparison"><h2>最终对比：两模型使用相同后处理</h2>']
    if policy == 'joint':
        chunks += ['<h3>当前后处理 pipeline</h3><ol>'] + ['<li>'+v+'</li>' for v in steps] + ['</ol>',
                   '<p>两个仿体不区分动静脉：先取血管并集，再清理、闭运算和填洞。面积/半径均按处理网格像素计；极小目标可设min_area=1，但下表固定使用4。当前互斥mask不会双重着色；绿色可单独显示整个血管并集。</p>']
    chunks += ['<p><b>以下 repaired/cleaned 作为本轮最终比较口径。</b>两模型使用相同图像、划分、预处理和后处理；Dice逐帧平均，双空记1。raw和旧largest结果仅用于诊断，不混入本表。正差值表示新模型更高。</p>',
               '<table><tr><th>数据集</th><th>类别</th><th>帧数</th>'+''.join('<th>'+esc(n)+' 最终Dice</th>' for n in names)+'<th>Δ Dice</th></tr>']
    rows = []
    index = {(r['dataset'], r['model'], r['processing'], r['vessel_class']):r for r in summary}
    for a in summary:
        if a['model'] != names[0] or a['processing'] != 'cleaned':
            continue
        b = index[(a['dataset'], names[1], 'cleaned', a['vessel_class'])]
        row = dict(dataset=a['dataset'], vessel_class=a['vessel_class'], n=a['n'],
                   baseline_model=names[0], candidate_model=names[1],
                   baseline_dice=a['dice'], candidate_dice=b['dice'], delta=b['dice']-a['dice'],
                   postprocessing=policy)
        rows.append(row)
        label = {'vein':'静脉','artery':'动脉','vessel':'血管并集'}[a['vessel_class']]
        chunks.append(f'<tr><td>{esc(a["dataset"])}</td><td>{label}</td><td>{a["n"]}</td><td>{a["dice"]:.4f}</td><td>{b["dice"]:.4f}</td><td>{row["delta"]:+.4f}</td></tr>')
    chunks += ['</table><p><a href="final_comparison.csv">下载最终对比表 CSV</a>。PMC测试仅一个受试者；当前规则由已浏览案例启发，结果是探索性对照，不代表完全独立的新测试。</p></section>']
    return '\n'.join(chunks), rows


def write_final_comparison(out, summary, names, policy):
    section, rows = final_comparison_section(summary, names, policy)
    with (out / 'final_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return section

def write_report(out, summary, gallery, names, split, policy="largest"):
    esc = html.escape
    chunks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
              '<title>U-Net checkpoint 对照评估</title><style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#172034;background:#f6f8fb}table{border-collapse:collapse;background:white;display:block;overflow-x:auto;max-width:100%}td,th{padding:8px 14px;border:1px solid #cbd5e1;text-align:right}th:first-child,td:first-child{text-align:left}figure{margin:24px 0;padding:16px;background:white;border:1px solid #d8e0ec;border-radius:12px}img{width:100%;height:auto}small{color:#526176}a{color:#096b8d}summary{cursor:pointer;font-weight:bold}code{background:#e8edf5;padding:2px 5px}</style>',
              '<h1>U-Net 同条件对照评估</h1>',
              f'<p>模型：{esc(names[0])} → {esc(names[1])}。划分：<b>{split}</b>。两版读取相同缓存、相同归一化，无增强；AMP 推理、argmax 决策。该页使用选定的最佳 checkpoint，不使用最后一轮参数。</p>',
              '<p>先看各域指标，再看静脉存在/缺失分组，最后打开同一张图的预测对照。颜色：<b style="color:#3282ff">蓝色静脉</b>、<b style="color:#f54141">红色动脉</b>、<b style="color:#219354">绿色未分型血管</b>。Mendeley 只评动脉，图中预测静脉无真值可供评分。</p>',
              '<p>每类 Dice 按帧平均，真值与预测均为空记 1。后处理固定为每类保留最大 8 连通域、最小面积 1 像素、不做形态学；仿体在血管并集上清理。没有用本轮测试结果调阈值。孤岛筛选不能判断唯一的小区域是否为误报。</p>',
              '<p><a href="summary.json">汇总及复现信息</a> · <a href="per_frame.csv">逐帧指标 CSV</a></p>']
    chunks.insert(3, write_final_comparison(out, summary, names, policy))
    index = {(r['dataset'], r['model'], r['processing'], r['vessel_class']): r for r in summary}
    keys = list(dict.fromkeys((r['dataset'], r['vessel_class']) for r in summary))
    for processing in (('raw', 'largest', 'cleaned') if policy == 'joint' else ('raw', 'cleaned')):
        chunks += [f'<h2>{processing}：{"原始预测" if processing == "raw" else ("旧规则：每类只留最大区域" if processing == "largest" or policy == "largest" else "新规则：区域重归类、闭运算与填洞")}</h2>',
                   '<table><tr><th>数据集 / 类别</th><th>帧数</th>'+''.join(f'<th>{esc(n)}</th>' for n in names)+'<th>Δ Dice</th></tr>']
        for dataset, cls in keys:
            a, b = [index[(dataset, n, processing, cls)] for n in names]
            chunks.append(f'<tr><td>{dataset} / {cls}</td><td>{a["n"]}</td><td>{a["dice"]:.4f}</td><td>{b["dice"]:.4f}</td><td>{b["dice"]-a["dice"]:+.4f}</td></tr>')
        chunks.append('</table>')
    chunks += ['<h2>3. 可见血管与缺失血管分开检查</h2><p>下表为固定后处理结果。缺失类误报按“任意一个预测像素”计，必须结合误报面积一起解读。vessel 缺失表示血管并集为空；PMC / Mus-V 的该行即全背景帧。</p>',
               '<table><tr><th>数据集 / 类别</th><th>可见帧数</th><th>旧→新 可见帧 Dice</th><th>缺失帧数</th><th>旧→新 误报帧数</th><th>旧→新 缺失帧平均误报像素</th></tr>']
    for dataset, cls in keys:
        a, b = [index[(dataset, n, 'cleaned', cls)] for n in names]
        fmt = lambda value: '—' if value is None else f'{value:.4f}'
        chunks.append(f'<tr><td>{dataset} / {cls}</td><td>{a["present_n"]}</td><td>{fmt(a["present_dice"])} → {fmt(b["present_dice"])}</td><td>{a["absent_n"]}</td><td>{a["absent_fp_n"]} → {b["absent_fp_n"]}</td><td>{fmt(a["absent_mean_fp_pixels"])} → {fmt(b["absent_mean_fp_pixels"])}</td></tr>')
    chunks += ['</table><h2>4. 同帧预测对照</h2><p>PMC 展示本划分全部帧。其他域按可评类别平均 Dice 的变化，确定性选取提升最大、退步最大、中位及均匀索引样本；这是诊断选图，不是随机抽样的总体证据。点击展开，每行依次为原图、真值、旧版原始/清理、新版原始/清理；点击图片查看原尺寸。</p>']
    for g in gallery:
        chunks.append(f'<details><summary>{esc(g["dataset"])} / {esc(g["sample"])} · {esc(g["reason"])} · Δ={g["delta"]:+.4f}</summary><figure><a href="{g["path"]}"><img loading="lazy" src="{g["path"]}"></a><figcaption>缓存索引 {g["index"]}；旧版分数 {g["scores"][0]:.4f} → 新版 {g["scores"][1]:.4f}。此处选图分数为可评 A/V 的均值，或单一可评类别的 Dice。</figcaption></figure></details>')
    chunks += ['<h2>5. 解释边界</h2><p>PMC 测试仅一个受试者的 30 帧，不能当作 30 个独立个体。Mus-V 按序列划分，未证实患者独立。v10 的历史 PRIMARY 与 v11 使用不同数据域和权重，不能直接对比。新增 PMC、采样配比及训练随机性同时变化，不能把差异单独归因于某一个因素。测试用于报告结果，后续模型选择应回到训练/验证划分。</p></html>']
    if policy == 'joint':
        chunks.insert(3, '<p><b>后处理探索性实验：</b>根据已浏览的测试案例提出修复规则，先在验证集检查，再对测试集作诊断。不能视为从未接触测试数据的独立泛化证明。修复不保证类别完全正确。</p>')
        chunks = [c.replace('后处理固定为每类保留最大 8 连通域、最小面积 1 像素、不做形态学；仿体在血管并集上清理。没有用本轮测试结果调阈值。', '同时报告旧最大区域规则（min_area=1）和新修复规则（min_area=4，closing_radius=2，填洞；区域归类占比≥0.6且只有一个类别锚点）。仿体仅清理并集；新区域不覆盖另一类保留像素。')
                   .replace('旧版原始/清理、新版原始/清理', '旧版原始/旧清理/新修复、新版原始/旧清理/新修复') for c in chunks]
    (out / 'index.html').write_text('\n'.join(chunks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs=2, required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', default=['musv', 'phantom_taobao', 'customer_3d_phantom', 'mendeley', 'pmc9883282'])
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--postprocess', choices=['largest', 'joint'], default='joint')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new output directory to preserve earlier results')
    if check_stale(args.datasets):
        parser.error('Dataset caches are stale; no evaluation performed')
    names = [p.stem for p in args.checkpoints]
    if len(set(names)) != 2:
        parser.error('Checkpoint basenames must be distinct')
    pick_idle_gpu()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models, provenance = {}, {}
    for name, path in zip(names, args.checkpoints):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        cfg = saved['config']
        if cfg.get('num_classes') != 3:
            parser.error('Only three-class A/V checkpoints are supported')
        model = UNet(1, 3, bilinear=cfg.get('bilinear', False), base_ch=cfg.get('base_ch', 64), dropout=cfg.get('dropout', 0))
        model.load_state_dict(saved['model'])
        models[name] = model.to(device, memory_format=torch.channels_last).eval()
        provenance[name] = dict(path=str(path), sha256=sha256(path), epoch=saved['epoch'], best_validation_score=saved['best_score'], config=cfg)
        del saved
    args.output.mkdir(parents=True)
    (args.output / 'images').mkdir()
    rows, gallery, caches = [], [], {}
    for ds_name in args.datasets:
        dataset = ReadDataset(args.split, dataset_name=ds_name, augment=False)
        caches[ds_name] = dict(path=str(dataset.data_path), sha256=sha256(dataset.data_path), n=len(dataset), label_mode=dataset.label_mode)
        labels = [f'cache_{i:04d}' for i in range(len(dataset))]
        if dataset.cfg['source_type'] == 'reviewed_csv':
            pipeline = DataPipeline(ds_name)
            source_rows = pipeline._load_reviewed_rows()['val' if args.split == 'validation' else args.split]
            assert len(source_rows) == len(dataset)
            # Verify cache-to-filename identity; never attach a filename by order alone.
            for i, row in enumerate(source_rows):
                im, mask = pipeline._load_and_crop(*pipeline._resolve_paths(row))
                assert np.array_equal(im, dataset.images[i]) and np.array_equal(mask, dataset.masks[i]), 'Reviewed source/cache mismatch'
                labels[i] = row['filename']
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
        predictions = {}
        scores = {name: [] for name in names}
        for name, model in models.items():
            predictions[name] = np.empty_like(dataset.masks)
            offset = 0
            with torch.inference_mode():
                for images, _, _ in loader:
                    with torch.autocast(device.type, enabled=device.type == 'cuda'):
                        pred = model(images.to(device, memory_format=torch.channels_last)).float().argmax(1).cpu().numpy().astype('uint8')
                    predictions[name][offset:offset + len(pred)] = pred
                    offset += len(pred)
            for i, pred in enumerate(predictions[name]):
                variants = [('raw', pred), ('cleaned', cleaned(pred, dataset.label_mode, args.postprocess))]
                if args.postprocess == 'joint':
                    variants.insert(1, ('largest', cleaned(pred, dataset.label_mode)))
                for processing, p in variants:
                    metrics = frame_metrics(p, dataset.masks[i], dataset.label_mode)
                    for metric in metrics:
                        rows.append(dict(dataset=ds_name, index=i, sample=labels[i], model=name, processing=processing, **metric))
                    if processing == 'cleaned':
                        selected = [m['dice'] for m in metrics if dataset.label_mode != 'full3' or m['vessel_class'] != 'vessel']
                        scores[name].append(float(np.mean(selected)))
            print(f'{name} {ds_name}: {len(dataset)} frames evaluated', flush=True)
        delta = np.asarray(scores[names[1]]) - np.asarray(scores[names[0]])
        order = np.argsort(delta, kind='stable')
        selections = {}
        if dataset.cfg['source_type'] == 'reviewed_csv':
            selections = {i: '全部人工测试帧' for i in range(len(dataset))}
        else:
            for i in order[:2]: selections[int(i)] = '退步最大 / 提升最小'
            for i in order[-2:]: selections[int(i)] = '提升最大 / 退步最小'
            selections[int(order[len(order)//2])] = '变化中位'
            for i in np.linspace(0, len(dataset)-1, min(3,len(dataset)), dtype=int):
                selections.setdefault(int(i), '均匀索引')
        for i, reason in selections.items():
            path = f'images/{ds_name}_{i:04d}.png'
            save_panel(args.output / path, dataset.images[i], dataset.masks[i], {n:predictions[n][i] for n in names}, dataset.label_mode, names, args.postprocess)
            gallery.append(dict(dataset=ds_name, index=i, sample=labels[i], reason=reason, delta=float(delta[i]), scores=[scores[n][i] for n in names], path=path))
        del predictions, dataset, loader
    summary = aggregate(rows)
    with (args.output / 'per_frame.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(dict(created=datetime.now().astimezone().isoformat(), split=args.split, torch_version=torch.__version__, checkpoint=provenance, caches=caches, protocol=dict(amp=device.type == 'cuda', grid='cached input grid', batch=args.batch, postprocessing=args.postprocess, parameters=(dict(min_area=4, closing_radius=2, fill_holes=True, dominance=.6) if args.postprocess == 'joint' else dict(min_area=1))), metrics=summary, gallery=gallery), indent=2, ensure_ascii=False))
    write_report(args.output, summary, gallery, names, args.split, args.postprocess)
    print(f'Report: {args.output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
