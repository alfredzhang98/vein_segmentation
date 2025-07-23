import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ----------------- 配置区 -----------------
test_id = "1"
test_name = "phantom_taobao"
base_dir       = Path("data") / test_name
images_dir     = base_dir / "images"
masks_dir      = base_dir / "masks"
aug_images_dir = base_dir / "images_aug"
aug_masks_dir  = base_dir / "masks_aug"

meta_file      = base_dir / f"meta_{test_name}_{test_id}.csv"
out_csv        = base_dir / f"meta_{test_name}_{test_id}_augmented.csv"

# 要增强的类别（现在是数组）
aug_types = ["negative", "positive"]

# 每张图像增强次数
N_AUG = 10
# ------------------------------------------------

# 确保输出目录存在
aug_images_dir.mkdir(parents=True, exist_ok=True)
aug_masks_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(meta_file)

# 只挑选需要增强的类别
to_aug_df = df[df["mask_status"].isin(aug_types)].copy()

def find_path(rel_path: str, search_dirs):
    """
    从多个目录中查找相对路径对应的文件。
    返回第一个存在的绝对路径，找不到时返回 None。
    """
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    for d in search_dirs:
        cand = d / rel_path
        if cand.exists():
            return cand
        cand2 = d / p.name
        if cand2.exists():
            return cand2
    return None

def random_transform(img, mask):
    """简单的几何 + 光照增强：翻转、旋转、亮度/对比度、噪声。"""
    # 水平翻转
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    # 随机旋转
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # 亮度 / 对比度
    alpha = random.uniform(0.9, 1.1)
    beta  = random.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 高斯噪声
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, mask

augmented_rows = []

# 遍历每个类别（也可直接对 to_aug_df 遍历，这里显示写出以满足“遍历 aug_type”的要求）
for cls in aug_types:
    subset = to_aug_df[to_aug_df["mask_status"] == cls]

    for idx, row in subset.iterrows():
        tmp_relative_path = row["relative_path"]
        tmp_mask_path     = row["mask_path"]

        if sys.platform.startswith("linux"):
            tmp_relative_path = tmp_relative_path.replace("\\", "/")
            tmp_mask_path     = tmp_mask_path.replace("\\", "/")

        img_path  = find_path(tmp_relative_path, [images_dir, base_dir])
        mask_path = find_path(tmp_mask_path,    [masks_dir,  base_dir])

        if img_path is None:
            print(f"[Warning] 找不到图像: {row['relative_path']} (row {idx})")
            continue
        if mask_path is None:
            print(f"[Warning] 找不到 mask: {row['mask_path']} (row {idx})")
            continue

        img = cv2.imread(str(img_path))
        m   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"[Warning] 读取 mask 失败: {mask_path}")
            continue
        mask = (m > 127).astype(np.uint8)

        for i in range(N_AUG):
            aug_img, aug_mask = random_transform(img, mask)
            aug_mask = (aug_mask * 255).astype(np.uint8)

            stem      = img_path.stem
            img_name  = f"{stem}_aug{i}.png"
            mask_name = f"{stem}_mask_aug{i}.png"

            cv2.imwrite(str(aug_images_dir / img_name), aug_img)
            cv2.imwrite(str(aug_masks_dir  / mask_name), aug_mask)

            new_row = row.copy()
            new_row["id"]            = f"{row['id']}_aug{i}"
            new_row["relative_path"] = str(aug_images_dir / img_name)
            new_row["mask_path"]     = str(aug_masks_dir  / mask_name)
            # 保持原有类别，不要写成整个数组
            new_row["mask_status"]   = cls

            augmented_rows.append(new_row)

# 合并原始 + 增强数据
final_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
final_df.to_csv(out_csv, index=False, encoding="utf-8")

print(f"增强完成，共生成 {len(augmented_rows)} 张样本 -> 保存至 {out_csv}")
