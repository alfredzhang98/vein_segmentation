# unet_inferencer.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2

try:
    from PySide6 import QtGui
    QImage = QtGui.QImage
except ImportError:
    QImage = None

try:
    from .unet import UNet
except ImportError:
    from unet import UNet

class UNetInferencer:
    def __init__(self, ckpt_path, device=None, cfg_override=None):
        """
        初始化模型推理器
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = torch.load(ckpt_path, map_location=self.device)

        self.cfg = dict(num_channel=1, num_classes=1, bilinear=False, amp=True, mean=(0.5,), std=(0.5,))
        self.cfg.update(self.ckpt.get("config", {}))
        if cfg_override:
            self.cfg.update(cfg_override)

        self.model = UNet(n_channels=1, n_classes=self.cfg["num_classes"], bilinear=self.cfg["bilinear"])
        self.model.to(self.device, memory_format=torch.channels_last)
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()

    def _normalize(self, img_np: np.ndarray) -> np.ndarray:
        img = img_np.astype(np.float32) / 255.0
        return (img - self.cfg["mean"][0]) / self.cfg["std"][0]

    def _denormalize(self, img_norm : np.ndarray) -> np.ndarray:
        return np.clip((img_norm * self.cfg["std"][0] + self.cfg["mean"][0]) * 255.0, 0, 255).astype(np.uint8)
    
    @staticmethod
    def convert_image_input_to_array(image_data):
        """
        Convert various image data formats to a numpy ndarray.

        Supported input types:
            - numpy.ndarray: returned as-is
            - PIL.Image.Image
            - bytes or bytearray
            - file paths (str or os.PathLike)
            - file-like objects with a read() method
            - QtGui.QImage with Format_Grayscale8

        Returns:
            A numpy.ndarray representing the image.
            For grayscale images, returns a 2D uint8 array of shape (H, W).
        """
        # If already a numpy array
        if isinstance(image_data, np.ndarray):
            return image_data

        # PIL Image
        if isinstance(image_data, Image.Image):
            return np.array(image_data)

        # Raw bytes or bytearray
        if isinstance(image_data, (bytes, bytearray)):
            buf = BytesIO(image_data)
            pil_img = Image.open(buf)
            return np.array(pil_img)

        # File-like object
        if hasattr(image_data, 'read'):
            pil_img = Image.open(image_data)
            return np.array(pil_img)

        # File path
        if isinstance(image_data, (str, os.PathLike)):
            pil_img = Image.open(str(image_data))
            return np.array(pil_img)

        # Qt QImage: only Format_Grayscale8 supported
        if QImage is not None and isinstance(image_data, QImage):
            qimg = image_data
            # Ensure it's 8-bit grayscale
            if qimg.format() != QImage.Format_Grayscale8:
                qimg = qimg.convertToFormat(QImage.Format_Grayscale8)

            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            return arr.reshape((height, width))

        raise TypeError(f"Unsupported image_data type: {type(image_data)}")

    def predict_array(self, image_array: np.ndarray, target_size=(576, 544), threshold=0.5, show=False, return_prob=False):
        """
        从字节数据中预测
        """
        orig_w, orig_h = image_array.shape[1], image_array.shape[0]
        img_np = image_array
    
            # 保存原始图像尺寸
        H, W = orig_h, orig_w
        
        # To ensure the image is in the correct format and shape
        # we can resize the image to the target size range
        trans = A.Compose([
            A.LongestMaxSize(max_size=min(target_size), p=1.0),
            # padding to ensure the image is same to target size            
            A.PadIfNeeded(
                min_height=target_size[0],
                min_width=target_size[1],
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0
            ),
            A.Normalize(mean=self.cfg["mean"], std=self.cfg["std"], max_pixel_value=255.0, p=1.0)
        ])

        # 2. 应用变换
        transformed = trans(image=img_np)
        img_transformed = transformed["image"]
        
        # 转换为tensor
        tensor = torch.from_numpy(img_transformed).unsqueeze(0).unsqueeze(0).float().to(self.device, memory_format=torch.channels_last)

        # 3. 推理
        with torch.inference_mode():
            use_amp = self.cfg.get("amp", True) and (self.device.type == "cuda")
            with torch.autocast(device_type=self.device.type if self.device.type != "mps" else "cpu", enabled=use_amp):
                logits = self.model(tensor)
                if self.cfg["num_classes"] == 1:
                    prob = torch.sigmoid(logits)[0, 0]
                    pred_np = prob.cpu().numpy()
                    if not return_prob:
                        pred_np = (pred_np >= threshold).astype(np.uint8) * 255
                else:
                    prob = torch.softmax(logits, dim=1)[0]
                    pred_np = prob.argmax(0).cpu().numpy().astype(np.uint8)

        # 4. 裁剪回原尺寸 (移除padding)
        scale = min(target_size) / max(H, W)
        h1, w1 = int(H * scale), int(W * scale)
        pad_h = target_size[0] - h1
        pad_w = target_size[1] - w1
        top  = pad_h // 2
        left = pad_w // 2
        pred_np = pred_np[top : top + h1, left : left + w1]

        # 5. 可视化
        if show:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原图
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            # 预测mask
            if self.cfg["num_classes"] == 1:
                axes[1].imshow(pred_np, cmap="gray")
            else:
                axes[1].imshow(pred_np)
            axes[1].set_title("Prediction Mask")
            axes[1].axis("off")
            
            # 叠加显示
            # 将原图转换为RGB格式
            img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
            
            # 需要将pred_np调整回原图大小
            pred_resized = cv2.resize(pred_np.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 创建绿色mask
            if self.cfg["num_classes"] == 1:
                # 对于二分类，使用阈值创建mask
                if return_prob:
                    mask = pred_resized > 0.5
                else:
                    mask = pred_resized > 127  # 因为之前乘以了255
            else:
                # 对于多分类，非背景区域作为mask
                mask = pred_resized > 0
            
            # 创建叠加图像
            overlay = img_rgb.copy().astype(np.float32)
            overlay[mask, 1] = np.minimum(overlay[mask, 1] + 100, 255)  # 增加绿色通道
            
            # 使用alpha混合
            alpha = 0.3  # 透明度
            blended = img_rgb.astype(np.float32) * (1 - alpha) + overlay * alpha
            
            axes[2].imshow(blended.astype(np.uint8))
            axes[2].set_title("Original + Mask Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

        return pred_np

    def predict_png(self, image_path, target_size=(576, 544), threshold=0.5, show=False, return_prob=False):
        """
        从PNG图像文件中预测
        """
        pil_img = Image.open(image_path).convert("L")
        img_np = np.array(pil_img)
        return self.predict_array(img_np, target_size, threshold, show, return_prob)


# If main
if __name__ == "__main__":
    # Example usage
    inferencer = UNetInferencer("ml/checkpoints/best_model_v1.pth")
    pred = inferencer.predict_png("ml/test1.png", show=True)
    print("Prediction shape:", pred.shape)