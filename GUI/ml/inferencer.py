# unet_inferencer.py
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
from .unet import UNet

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

    def _normalize(self, img_np):
        img = img_np.astype(np.float32) / 255.0
        return (img - self.cfg["mean"][0]) / self.cfg["std"][0]

    def _denormalize(self, img_norm):
        return np.clip((img_norm * self.cfg["std"][0] + self.cfg["mean"][0]) * 255.0, 0, 255).astype(np.uint8)

    def predict_png(self, image_path, target_size=(576, 544), threshold=0.5, show=False, return_prob=False):
        """
        单张图像推理
        """
        # 1. 读取图像并归一化
        pil_img = Image.open(image_path).convert("L")
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)
        img_norm = self._normalize(img_np)

        # 2. 转 tensor & padding
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        H, W = tensor.shape[2:]
        pad_bottom = max(target_size[0] - H, 0)
        pad_right  = max(target_size[1] - W, 0)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), value=0.0).to(self.device, memory_format=torch.channels_last)

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

        # 4. 裁剪回原尺寸
        pred_np = pred_np[:H, :W]

        # 5. 可视化
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            if self.cfg["num_classes"] == 1:
                axes[1].imshow(pred_np, cmap="gray")
            else:
                axes[1].imshow(pred_np)
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

        return pred_np
    
    def predict_PIL(self, image_data: PILImage, target_size=(576, 544), threshold=0.5, show=False, return_prob=False):
        """
        流式图像推理
        """
        # 1. 从字节数据创建 PIL 图像
        pil_img = image_data.convert("L")
        img_np = np.array(pil_img)
        img_norm = self._normalize(img_np)

        # 2. 转 tensor & padding
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        H, W = tensor.shape[2:]
        pad_bottom = max(target_size[0] - H, 0)
        pad_right  = max(target_size[1] - W, 0)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), value=0.0).to(self.device, memory_format=torch.channels_last)

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

        # 4. 裁剪回原尺寸
        pred_np = pred_np[:H, :W]

        # 5. 可视化
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            if self.cfg["num_classes"] == 1:
                axes[1].imshow(pred_np, cmap="gray")
            else:
                axes[1].imshow(pred_np)
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

        return pred_np
