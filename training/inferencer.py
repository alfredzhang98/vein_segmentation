# unet_inferencer.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
import os
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
        Initialize the model inferencer.
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
        if isinstance(image_data, np.ndarray):
            return image_data

        if isinstance(image_data, Image.Image):
            return np.array(image_data)

        if isinstance(image_data, (bytes, bytearray)):
            buf = BytesIO(image_data)
            pil_img = Image.open(buf)
            return np.array(pil_img)

        if hasattr(image_data, 'read'):
            pil_img = Image.open(image_data)
            return np.array(pil_img)

        if isinstance(image_data, (str, os.PathLike)):
            pil_img = Image.open(str(image_data))
            return np.array(pil_img)

        if QImage is not None and isinstance(image_data, QImage):
            qimg = image_data
            if qimg.format() != QImage.Format_Grayscale8:
                qimg = qimg.convertToFormat(QImage.Format_Grayscale8)

            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.frombuffer(ptr, np.uint8)
            return arr.reshape((height, width))

        raise TypeError(f"Unsupported image_data type: {type(image_data)}")

    def predict_array(self, image_array: np.ndarray, target_size=(576, 544),
                       threshold=0.5, return_prob=False):
        """
        Predict vessel segmentation on a grayscale image.

        Preprocessing matches training eval_transform: resize to target_size,
        normalize to [-1, 1] (mean=0.5, std=0.5).

        Args:
            image_array: grayscale uint8 ndarray (H, W)
            target_size: (H, W) matching training, default (576, 544)
            threshold:   binarization threshold
            return_prob: True returns probability map [0,1], False returns binary mask {0,255}

        Returns:
            pred_np: ndarray with same size as target_size
        """
        img_resized = cv2.resize(image_array, (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_AREA)

        mean_val = self.cfg["mean"][0] if isinstance(self.cfg["mean"], (list, tuple)) else self.cfg["mean"]
        std_val = self.cfg["std"][0] if isinstance(self.cfg["std"], (list, tuple)) else self.cfg["std"]
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean_val) / std_val

        tensor = (torch.from_numpy(img_norm)
                  .unsqueeze(0).unsqueeze(0).float()
                  .to(self.device, memory_format=torch.channels_last))

        with torch.inference_mode():
            use_amp = self.cfg.get("amp", True) and (self.device.type == "cuda")
            autocast_dev = self.device.type if self.device.type != "mps" else "cpu"
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                logits = self.model(tensor)

            if self.cfg["num_classes"] == 1:
                prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
                pred_np = prob if return_prob else (prob >= threshold).astype(np.uint8) * 255
            else:
                prob = torch.softmax(logits.float(), dim=1)[0]
                pred_np = prob.argmax(0).cpu().numpy().astype(np.uint8)

        return pred_np

    def predict_png(self, image_path, target_size=(576, 544), threshold=0.5, return_prob=False):
        """Predict from an image file (auto-converts to grayscale)."""
        img_np = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_np is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        return self.predict_array(img_np, target_size, threshold, return_prob)


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/best_model.pth"
    img_path = sys.argv[2] if len(sys.argv) > 2 else "test.png"
    inferencer = UNetInferencer(ckpt)
    pred = inferencer.predict_png(img_path)
    print(f"Prediction shape: {pred.shape}, unique: {np.unique(pred)}")
