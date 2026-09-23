"""Summarize the v11 refresh independently from temporal-model experiments."""
import json
import os
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mistune

RUN = Path('results/unet/runs/v11_refresh_20260923')
OUT = Path('results/unet/generated/v11_refresh')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decision=json.loads((RUN/'decision.json').read_text())
    text='''# v11 单帧重训结果

2026-09-23。沿用五域 partition CE＋Dice，从原 v11 微调；本轮不使用时序模型。

**新候选未替换原 v11：PMC 静脉改善，但 Mus-V 静脉退化。**

[可浏览报告](generated/v11_refresh/index.html) · [Mus-V 原图、标注与新旧 v11 视频](generated/musv_review/index.html) · [训练日志](runs/v11_refresh_20260923/history.jsonl)

## 同条件比较

同一清理后划分、同一 joint 后处理，逐帧 Dice 平均；双空 Dice 记 1。历史测试已多次用于开发，不作为新的独立泛化证据。

| 划分 | 数据集 | 类别 | 帧数 | 原 v11 | 新候选 | 差值 |
|---|---|---|---:|---:|---:|---:|
'''
    for split in ['validation','test']:
        metrics=json.loads((RUN/f'{split}_comparison/summary.json').read_text())['metrics']
        lookup={(r['dataset'],r['model'],r['vessel_class']):r for r in metrics if r['processing']=='cleaned'}
        for domain,classes in [('musv',['vein','artery']),('pmc9883282',['vein','artery']),('mendeley',['artery']),('phantom_taobao',['vessel']),('customer_3d_phantom',['vessel'])]:
            for cls in classes:
                old,new=lookup[domain,'unet_v11',cls],lookup[domain,'candidate',cls]
                text+=f'| {split} | {domain} | {cls} | {old["n"]} | {old["dice"]:.4f} | {new["dice"]:.4f} | {new["dice"]-old["dice"]:+.4f} |\n'
    text+='''
## 训练与局限

正式单卡 bfloat16 微调完成 5 个 epoch，验证选中 epoch 1。首次多卡非有限梯度运行已判无效，移至项目外历史归档，不计作训练收益。

PMC 标注从 362 张中排除 44 张无接触帧，保留 318 张（训练 264、验证 28、测试 26）。有效组织回声中的空血管标签仍保留。[数据审计](../../data/audits/20260923_contact_cleanup/contact_audit.json)。

本轮不仅增加标签，也改变了采样、增强和选模权重：PMC 训练采样占比约 19%→24%，验证选模权重 10%→30%。新增连续片段主要来自同一训练受试者。因此当前结果不能单独归因于新增标签好坏；下一次应固定其余设置比较旧/新标注，并在验证选模时限制各域退化。

![验证曲线](generated/v11_refresh/training.png)

## 文件与复现

- 模型代码：`models/unet/`；训练配置：`configs/v11_refresh.env`。
- 本地实验记录：`results/unet/runs/v11_refresh_20260923/`，整目录不上传 Git。
- [验证评估页面](runs/v11_refresh_20260923/validation_comparison/index.html) · [测试评估页面](runs/v11_refresh_20260923/test_comparison/index.html) · [替换决策](runs/v11_refresh_20260923/decision.json)。
- 活动权重仍是 `models/unet/checkpoints/unet_v11.pth`；新候选已归档，未覆盖默认。
- [时序模型结果](../temporal/RESULTS_CN.md) 单独放在 `results/temporal/`。

重新生成本摘要与页面：`python -m results.unet.summarize_refresh`。
'''
    text+=f'\n原 v11 SHA256：`{decision["retained_sha256"]}`。\n'
    Path('results/unet/V11_REFRESH_CN.md').write_text(text)
    history=[json.loads(l) for l in (RUN/'history.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['epoch'] for r in history],[r['loss'] for r in history],marker='o');axes[0].set_title('Training loss')
    for domain in ['musv','pmc9883282']:
        axes[1].plot([r['epoch'] for r in history],[r['validation'][domain]['per_class']['dice_vein'] for r in history],marker='o',label=domain)
    axes[1].set_title('Validation vein Dice (raw)');axes[1].legend()
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training.png',dpi=150);plt.close(fig)
    body=mistune.create_markdown(plugins=['table'])(text)
    def rebase(match):
        prefix,url,suffix=match.groups()
        if ':' in url or url.startswith('#'):return match.group(0)
        return prefix+os.path.relpath(Path('results/unet')/url,OUT)+suffix
    body=re.sub(r'(href="|src=")([^"]+)(")',rebase,body)
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v11 单帧重训结果</title><style>body{max-width:1250px;margin:24px auto;padding:0 18px;font:16px/1.8 sans-serif}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccc;padding:6px 12px;white-space:nowrap}img{max-width:100%}a{color:#075ba4}</style>'+body+'</html>')
    print('U-Net refresh report generated')

if __name__=='__main__':main()
