import os
import pandas as pd
import numpy as np
import cv2
import random
from pathlib import Path

# 配置路径
base_dir       = Path("data") / "phantom_taobao"
images_dir     = base_dir / "images"
masks_dir      = base_dir / "masks"
aug_images_dir = base_dir / "images_aug"
aug_masks_dir  = base_dir / "masks_aug"

# 确保输出目录存在
aug_images_dir.mkdir(parents=True, exist_ok=True)
aug_masks_dir.mkdir(parents=True, exist_ok=True)

# 读取 CSV
meta_file = base_dir / "meta_phantom_taobao_1.csv"
df = pd.read_csv(meta_file)

# 只增强标注为 positive 的样本
pos_df = df[df['mask_status'] == 'positive'].copy()

# 每张图像增强次数
N_AUG = 10

def find_path(rel_path: str, search_dirs):
    """
    从给定的多个目录中查找相对路径对应的文件。
    返回第一个存在的绝对路径，找不到时返回 None。
    """
    p = Path(rel_path)
    # 如果 rel_path 已是绝对路径并存在，则直接返回
    if p.is_absolute() and p.exists():
        return p
    # 否则在各搜索目录下尝试
    for d in search_dirs:
        cand = d / rel_path
        if cand.exists():
            return cand
        # 有时 CSV 中存的是以目录开头的 `phantom_taobao/...`，尝试拿最后一级文件名
        cand2 = d / p.name
        if cand2.exists():
            return cand2
    return None

def random_transform(img, mask):
    """简单几何与光强增强：翻转、旋转、亮度/对比度、噪声。"""
    # 水平翻转
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    # 随机旋转
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    # 亮度/对比度
    alpha = random.uniform(0.9, 1.1)
    beta  = random.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # 高斯噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, mask

augmented_rows = []

for idx, row in pos_df.iterrows():
    # 查找原始图和 mask 的绝对路径
    img_path  = find_path(row['relative_path'], [images_dir, base_dir])
    mask_path = find_path(row['mask_path'],    [masks_dir,  base_dir])

    if img_path is None:
        print(f"[Warning] 找不到图像: {row['relative_path']} (row {idx})")
        continue
    if mask_path is None:
        print(f"[Warning] 找不到 mask: {row['mask_path']} (row {idx})")
        continue

    img  = cv2.imread(str(img_path))
    m    = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"[Warning] 读取 mask 失败: {mask_path}")
        continue
    mask = (m > 127).astype(np.uint8)

    for i in range(N_AUG):
        aug_img, aug_mask = random_transform(img, mask)
        aug_mask = (aug_mask * 255).astype(np.uint8)

        # 文件名
        stem      = img_path.stem
        img_name  = f"{stem}_aug{i}.png"
        mask_name = f"{stem}_mask_aug{i}.png"

        # 保存增强样本
        cv2.imwrite(str(aug_images_dir / img_name), aug_img)
        cv2.imwrite(str(aug_masks_dir  / mask_name), aug_mask)

        # 记录新的 CSV 行
        new_row = row.copy()
        new_row['id']            = f"{row['id']}_aug{i}"
        new_row['relative_path'] = str(aug_images_dir / img_name)
        new_row['mask_path']     = str(aug_masks_dir  / mask_name)
        new_row['mask_status']   = 'positive'
        augmented_rows.append(new_row)

# 合并原始 + 增强数据并保存到新 CSV
aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
out_csv = base_dir / "meta_phantom_taobao_1_augmented.csv"
aug_df.to_csv(out_csv, index=False, encoding='utf-8')
print(f"增强完成，共生成 {len(augmented_rows)} 张样本 -> 保存至 {out_csv}")
