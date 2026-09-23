"""Build a source-backed durable pipeline report from completed runs."""
from pathlib import Path
import json,html,hashlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mistune
from models.temporal.evaluate import motion_diagnostics
U=Path('results/unet/runs/v11_refresh_20260923');S=Path('results/temporal/runs/convgru_20260923');D=Path('data/audits/20260923_contact_cleanup');OUT=Path('results/temporal/generated/report')
def read(p):return json.loads(Path(p).read_text())
def history(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def values(data):return {(r['dataset'],r['model'],r['class_id']):r for r in data['summary']}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    vdecision=read(U/'decision.json');choice=read(S/'dense_pre_test_selection.json');baseline=read(S/'evaluation/baseline_joint/summary.json');candidate=read(S/'evaluation/candidate_joint/summary.json');ablation=read(S/'evaluation/no_history/summary.json')
    b,c,a=map(values,[baseline,candidate,ablation]);fail=[]
    for d in ['musv','pmc9883282']:
        for k in [1,2]:
            if c[d,'current',k]['dice']<b[d,'current',k]['dice']-.005:fail.append(f'{d}/{k}: current Dice')
            if c[d,'current',k]['false_positive_frames']>b[d,'current',k]['false_positive_frames']+1:fail.append(f'{d}/{k}: absent false positives')
            if c[d,'forecast',k]['dice']<max(b[d,'flow',k]['dice'],b[d,'persistence',k]['dice'])-.005:fail.append(f'{d}/{k}: forecast vs simple baselines')
    bc={(r['dataset'],r['model'],r['class_id']):r for r in baseline['continuity']};cc={(r['dataset'],r['model'],r['class_id']):r for r in candidate['continuity']}
    for d in ['musv','pmc9883282']:
        for metric in ['jumps20','area_jumps50']:
            if sum(cc[d,'current',k][metric] for k in [1,2])>=sum(bc[d,'current',k][metric] for k in [1,2]):fail.append(f'{d}: no reduction in {metric}')
    passed=choice['passed_validation_gates'] and not fail
    decision=dict(accepted=passed,passed_validation=choice['passed_validation_gates'],test_failures=fail,checkpoint_sha256=choice['sha256'],epoch=choice['epoch'],protocol='fixed validation selection; reused development test, no independent generalization claim')
    (S/'decision.json').write_text(json.dumps(decision,indent=2))
    text='# 时序跟踪与预测：双尺度 ConvGRU 实验结果\n\n2026-09-23。\n\n'
    text+='**双尺度 ConvGRU 已训练并完成完整序列对照；空间基线仍为原 v11。** '+('Stage1 达到本轮预设探索性门槛。' if passed else '**Stage1 尚未达到整体质量、连续性和预测的验收要求，不作为已改善的默认模型。**')+'\n\n'
    text+='[完整 8fps 视频与逐帧面积](generated/video/index.html) · [交互浏览本报告](generated/report/index.html) · [数据处理与设计记录](runs/convgru_20260923/EXPERIMENT_NOTES_CN.md)\n\n'
    text+='## 数据清理与实际监督\n\n362 张人工标注中排除 44 张无接触黑图，保留 **318 张：训练 264、验证 28、测试 26**。保留 3 张有有效组织回声的全空标签；暗但仍有组织的过渡帧不按空 mask 删除。原图与标签有归档，活动 CSV 用 pass 和 frame_valid=false 排除。全 PMC 5215 帧排除 505 帧，保留 4710；加上 Mus-V 3114，共 7824 个有效帧。其他四域未检出同规则的近空候选。\n\n'
    text+='筛选规则仅针对本地 PMC 裁剪：主体 ROI 均值 <0.5 且灰度>20 的比例 <1%，候选和边界样例已经看图复核。这不是跨设备接触检测器。无效帧切断序列并清空隐藏状态，不压缩时间、不跨缺口预测。所有分割和时序对照均重新使用清理后的相同标签，不能直接对比旧报告包含黑图时的分数。\n\n'
    text+=table(['数据集','划分','有效帧','人工标签','连续段','相邻两帧都有标签'],[[r['dataset'],r['split'],r['frames'],r['labels'],r['segments'],r['paired_labels']] for r in read(D/'data_counts.json')])
    text+='\n最终训练窗口要求 8 帧输入及第 9 帧目标全部有人工标签：Mus-V **1571** 个、PMC **73** 个重叠窗口。PMC 训练的真实相邻标签对由此前 0 增至 156；重叠窗口不等于独立受试者。PMC 验证/测试仍没有密集相邻 GT，不能凭视频平滑就声称真实运动或延迟已经准确。\n\n'
    text+='## 单帧基线\n\n本模型从保留的原 v11 初始化。v11 重训候选没有跨域改善，未替换；单帧实验的训练曲线、全部域指标和视频统一见 [U-Net 重训结果](../unet/V11_REFRESH_CN.md)。\n\n'
    text+='## Stage1：架构、训练和选择\n\n在 U-Net 编码器 1/8、1/16 处加入 64 通道 ConvGRU，以零初始化投影保留初始单帧输出；保留完整解码器。因果逐帧更新状态，不等待未来图像。预测头从当前因果特征产生位移和有限类别调整，运输当前概率到下一帧。它是原始帧步长的一步预测，没有标定物理时间，也没有长期对象 ID 真值。\n\n'
    text+='当前/未来分割使用 full3 CE＋前景 Dice；几何项分别监督动静脉中心和面积；变化项监督相邻人工标签的真实增量。中心项只用于对应类别存在时，面积项允许真实消失。损失权重为当前 1、未来 0.7、几何 0.15、变化 0.2。所有 8 帧共享几何增强，PMC 数据只来自训练受试者 20。\n\n'
    text+='先做联合微调，再做冻结 U-Net 的采样对照，均未通过门槛。最终收紧为逐帧完整标注窗口，固定 U-Net 权重、BatchNorm 统计和 Dropout 状态，只训练时序模块与预测头。选择仅依据验证，epoch0 为未训练参考；未通过门槛时仍保存验证分数最佳的实际训练候选供诊断，不能把 epoch0 冒充训练收益。一次相同种子重跑只修正训练候选保存逻辑，不作为独立重复实验。\n\n'
    text+=f"最终研究候选 epoch **{choice['epoch']}**；验证质量门槛：**{'通过' if choice['passed_validation_gates'] else '未通过'}**。完整日志与各 epoch 验证统计见 `runs/convgru_20260923/training/`。\n\n"
    text+='![训练曲线](generated/report/training.png)\n\n'
    text+='## 同条件当前分割与下一帧预测\n\n下表均使用相同 joint 后处理；raw 指标保存在对应 `evaluation/candidate_raw` 和 `evaluation/baseline_raw` 目录。当前 Mus-V/PMC 标签数 470/26，未来标签数 457/25。双空 Dice 为 1；另报告空帧误报，避免用空帧得分掩盖漏检。\n\n'
    rows=[]
    for d in ['musv','pmc9883282']:
        for k,cl in [(1,'静脉'),(2,'动脉')]:rows.append([d,cl,*[f'{v:.4f}' for v in [b[d,'current',k]['dice'],c[d,'current',k]['dice'],b[d,'persistence',k]['dice'],b[d,'flow',k]['dice'],c[d,'forecast',k]['dice'],a[d,'current',k]['dice'],a[d,'forecast',k]['dice']]]])
    text+=table(['域','类别','v11 当前','Stage1 当前','复制 v11','v11 光流','Stage1 未来','去历史当前','去历史未来'],rows)
    text+='\n去历史保留学习过的网络和预测头，但每一帧都重置 ConvGRU 状态，用来检查历史是否实际产生作用；不是另训一套单帧模型。\n\n'
    text+='## 连续性、误报与跟踪延迟\n\n连续性阈值为质心位移 >20px、双非空面积比 >1.5；它们是代理指标，真实运动也可能超过阈值。所有无效帧已排除，相邻关系只在同一连续段内统计。\n\n'
    rows=[]
    for d in ['musv','pmc9883282']:
        for k,cl in [(1,'静脉'),(2,'动脉')]:
            for name,ct,mt in [('v11',bc,b),('Stage1',cc,c)]:
                r=ct[d,'current',k];m=mt[d,'current',k];rows.append([d,cl,name,r['both_present'],r['jumps20'],r['area_jumps50'],f"{r['appears']}/{r['disappears']}",f"{m['false_positive_frames']}/{m['absent_n']}",f"{m['class_confusion']:.4f}",f"{m['centroid_error']:.2f}"])
    text+=table(['域','类别','输出','双非空帧对','大位移','面积突变','出现/消失','空帧误报','串类像素率','质心误差px'],rows)
    text+='\n质心误差仅在 GT/预测均非空时定义，必须与漏检和空帧指标一起看。延迟诊断在相同密集标注支持上比较 -3 至 +3 帧的中心/面积误差，正值表示预测滞后；它不是经过物理时间标定的延迟。\n\n'
    lr=[]
    for name,res in [('v11',baseline),('Stage1',candidate)]:
        for d in ['musv','pmc9883282']:
            for k in [1,2]:
                eligible=[r for r in res['lag'] if r['dataset']==d and r['class_id']==k and r['estimable']];lr.append([name,d,'静脉' if k==1 else '动脉',len(eligible),statistics.median([r['best_lag_frames'] for r in eligible]) if eligible else '不可估计'])
    text+=table(['模型','域','类别','可估计序列数','最佳偏移中位数/帧'],lr)
    mr=[]
    for name,res in [('v11',baseline),('Stage1',candidate)]:
        for r in motion_diagnostics(res['traces']):
            mr.append([name,r['dataset'],r['class_id'],r['labelled_pairs'],r['missing_prediction_pairs'],f"{r['motion_error_px']:.3f}" if r['motion_error_px'] is not None else '不可估计',f"{r['log_area_change_error']:.3f}" if r['log_area_change_error'] is not None else '不可估计'])
    text+='\n下面直接比较预测变化与人工标签变化，避免把静止输出当成正确跟踪。缺失预测的帧对另计，不从误差均值中悄悄隐藏。\n\n'
    text+=table(['模型','域','类别ID','GT双非空相邻对','预测缺失对','位移变化误差px','log面积变化误差'],mr)
    text+='\nMus-V 测试人工静脉标签本身就有 25 个 >20px 的相邻位置变化、38 个 >1.5 倍面积变化。因此不能把阈值次数全部称为错误，更不能把“零变化”设为目标；标签变化也可能含标注误差，需要结合原图判断。\n\n'
    text+='\n## 已定位的问题与尚待验证的原因\n\n'
    text+='**历史状态在本轮模型中确实造成了额外退化。** Mus-V 静脉 Dice 在保留历史时为 0.5510，每帧重置历史后回升到 0.6077，原 v11 为 0.6409。这项对照定位到学习后的时序状态融合，不能把问题全部归因于标注不足；重置后仍未恢复到 v11，也说明单帧经过新增模块同样存在偏差。\n\n'
    text+='**当前训练与长视频推理的状态分布不一致。** 训练每个 8 帧窗口从空状态开始，完整视频则持续携带状态数百帧。这是代码中可确认的设计差异，但尚未单独证明它解释了多少退化。下一轮需要先比较不同历史长度，再测试带历史预热的截断反向传播，避免直接把模型变大。\n\n'
    text+='**预测没有超过简单基线。** 下一帧分数未达到复制当前分割或仅用过去图像的光流外推，说明当前位移＋类别残差头还没有学到有效的运动规律。当前/未来目标的梯度冲突、运动信息不足是待做消融的假设，并非已经证实的结论。PMC 验证与测试缺少连续人工标签则是明确的验收缺口。\n\n'
    text+='## 结论与下一步\n\n'
    if not passed:text+='本轮完成了数据清理、v11 候选重训和双尺度 ConvGRU 的实际训练，但没有满足整体验收，不能承诺不跳变或预测优于 v11 加简单运动外推。v11 保留原权重；Stage1 只保留一个明确标记为研究候选的权重供检查，不冒充已通过的部署升级。\n\n'
    text+='下一步应针对本轮可复现的失败分开处理：检查当前帧更新与未来损失的梯度冲突，用只训练预测头/只训练当前状态的对照定位；给预测头加入截至当前可观测的运动先验并检验残差学习是否超过光流；在未用于训练的独立受试者上补连续短片，用真实变化和消失/出现时刻验证延迟。优先解决可测的状态更新与转移问题，不继续盲目放大模型或重复同一人的孤立帧。上述是待验证方向，本轮未宣称已经实现。\n\n'
    text+='## 实现检查与保留产物\n\n本轮训练结束时 72 项项目测试通过；目录整理后保留的当前实现另经 44 项本地测试通过。16 个连续原生 PMC 帧的在线/缓存当前分割与未来预测完全一致；无接触会重置状态，重置后的首帧可复现。Stage1 内嵌的 U-Net 所有权重和归一化统计与保留的 v11 完全相同，验证了退化来自新增时序/预测部分。\n\n'
    text+='唯一活动时序权重为 `models/temporal/checkpoints/vessel_tracker_candidate.pth`，状态为研究候选、未通过。旧时序默认权重和本轮冗余训练 checkpoint 已归档，配置、日志、指标、视频保留；[权重归档映射](runs/convgru_20260923/provenance/checkpoint_archive.json) 可用于复核原路径。\n\n'
    text+='## 复现与来源\n\n```bash\npython -m models.temporal.prepare --output /tmp/NEW_STAGE1_CACHE\n# 复制 configs/temporal.json，修改 cache 和 run_dir，避免覆盖当前记录\nCUDA_VISIBLE_DEVICES=1 python -m models.temporal.train --config configs/YOUR_STAGE1.json\npython -m models.temporal.evaluate --checkpoint PATH --cache CACHE --output NEW_EVAL --joint\n```\n\n'
    text+=f"- 默认 v11 SHA256：`{vdecision['retained_sha256']}`\n- Stage1 研究候选 SHA256：`{choice['sha256']}`\n- 数据索引 SHA256：`{candidate['index_sha256']}`\n- [排除帧审计](../../data/audits/20260923_contact_cleanup/contact_audit.json) · [清理清单](runs/convgru_20260923/provenance/removed_obsolete_caches.json) · [v11 决策](../unet/runs/v11_refresh_20260923/decision.json) · [Stage1 决策](runs/convgru_20260923/decision.json)\n"
    Path('results/temporal/RESULTS_CN.md').write_text(text)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    comparisons=read(S/'comparison_history.json');comparisons['Frozen U-Net, dense labels']=history(S/'training/history.jsonl')
    for label,hs in comparisons.items():
        axes[0].plot([r['epoch'] for r in hs],[r['score'] for r in hs],marker='o',label=label);valid=[r for r in hs if r['epoch']];axes[1].plot([r['epoch'] for r in valid],[r['loss'] for r in valid],marker='o',label=label)
    axes[0].set_title('Temporal validation selection score');axes[1].set_title('Temporal training loss');axes[0].legend(fontsize=7)
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training.png',dpi=160);plt.close(fig)
    body=mistune.create_markdown(plugins=['table'])(text)
    import re,os
    def rebase(match):
        prefix,url,suffix=match.groups()
        if ':' in url or url.startswith('#'):return match.group(0)
        return prefix+os.path.relpath(Path('results/temporal')/url,OUT)+suffix
    body=re.sub(r'(href="|src=")([^"]+)(")',rebase,body)
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据清理与 Stage1 实验结果</title><style>body{max-width:1450px;margin:28px auto;padding:0 20px;font:16px/1.8 sans-serif;background:#f6f8fb;color:#183040}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:8px 12px;border:1px solid #ccd6e0;white-space:nowrap}img{max-width:100%}a{color:#075ba4}pre{overflow:auto;background:#e6edf4;padding:14px}</style>'+body+'</html>')
    print('REPORT COMPLETE',decision,flush=True)
if __name__=='__main__':main()
