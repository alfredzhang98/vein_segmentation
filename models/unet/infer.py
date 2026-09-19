"""
infer.py — deployment inference. Self-contained: only torch / numpy / cv2
plus `model.py`, so it can be copied into the robot project as-is.

WHAT THE OLD VERSION GOT WRONG
------------------------------
The previous predict_array() did:

    img_resized = cv2.resize(image_array, (544, 576))     # <- non-uniform resize
    ...
    return mask                                            # <- stays at 576x544

Two separate defects:

1. That resize scales height and width by DIFFERENT factors, so a round vessel is
   delivered to the network as an ellipse. Artery-vs-vein is a shape judgement (an
   artery is round and non-collapsible, a vein is flattened and compressible), so it
   corrupts the very cue the 3-class model relies on. It also no longer matches how
   the training arrays are built.

2. It returned the mask in 576x544 network space. Geometry was then measured there and
   converted with mm_per_px calibrated for the CROP frame — but the two axes had been
   scaled differently, so no single scalar could convert both. Measured on a real
   frame: lateral offset was +27.6% off, radius +17.3%.

WHAT THIS VERSION DOES
----------------------
    crop frame  --fit()-->  576x544  --model-->  mask  --unfit()-->  crop frame
                                                                     ^^^^^^^^^^
                                        geometry is measured HERE, where mm_per_px
                                        is defined. The model never sees a millimetre.

fit() only ever pads, centre-crops, or scales BOTH axes by one factor — aspect ratio is
never touched — and unfit() inverts it exactly. In the default 'croppad' mode there is
no scaling at all, so one network pixel IS one crop-frame pixel and the inverse is a
pure offset with zero resampling loss.

Usage
-----
    inf = UNetInferencer("checkpoint_mixav_v1_....pth")

    # returns the mask on the CROP-FRAME grid — same size as what you passed in
    mask = inf.predict(cropped_gray_frame)          # 0=bg, 1=vein, 2=artery

    # geometry, in mm, using the calibration for the current depth setting
    g = inf.measure(mask, mm_per_px=0.0832)         # 4 cm; use 0.104 for 5 cm
    print(g["artery"]["lateral_mm"], g["artery"]["depth_mm"], g["artery"]["radius_mm"])
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .model import UNet
except ImportError:
    from model import UNet

BG, VEIN, ARTERY = 0, 1, 2
CLASS_NAME = {VEIN: "vein", ARTERY: "artery"}


# ---------------------------------------------------------------------------
# fit / unfit  — MUST stay identical to dataPrepare.fit / dataPrepare.unfit_mask.
# Duplicated (not imported) only so this file has no training-side dependencies.
# tests/test_inferencer_parity.py asserts the two agree bit for bit.
# ---------------------------------------------------------------------------

def fit(img, mask, target_hw, mode="croppad"):
    """Cropped frame -> model input size, invertibly, without touching aspect ratio.

    croppad   : no scaling. Pad if smaller, centre-crop if larger.
    letterbox : uniform downscale by ONE factor, then pad.

    Vertical alignment is TOP (row 0 of a B-mode frame is the probe face, so the skin
    line stays at row 0); horizontal alignment is CENTRE (the probe axis is the middle
    of the frame, which is what lateral offset is measured from).
    """
    TH, TW = target_hw
    H, W = img.shape[:2]

    if mode == "letterbox":
        scale = min(TH / H, TW / W)
    elif mode == "croppad":
        scale = 1.0
    else:
        raise ValueError(f"Unknown fit mode: {mode!r} (croppad | letterbox)")

    if scale != 1.0:
        nh, nw = int(round(H * scale)), int(round(W * scale))
        img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_s = None if mask is None else cv2.resize(mask, (nw, nh),
                                                      interpolation=cv2.INTER_NEAREST)
    else:
        nh, nw, img_s, mask_s = H, W, img, mask

    sy = dy = 0                                  # top-anchored
    sx = max(0, (nw - TW) // 2)                  # centred horizontally
    dx = max(0, (TW - nw) // 2)
    ch, cw = min(nh, TH), min(nw, TW)

    img_o = np.zeros((TH, TW), dtype=img.dtype)
    img_o[dy:dy + ch, dx:dx + cw] = img_s[sy:sy + ch, sx:sx + cw]

    mask_o = None
    if mask_s is not None:
        mask_o = np.zeros((TH, TW), dtype=mask.dtype)
        mask_o[dy:dy + ch, dx:dx + cw] = mask_s[sy:sy + ch, sx:sx + cw]

    return img_o, mask_o, dict(mode=mode, scale=scale, H=H, W=W, nh=nh, nw=nw,
                               sy=sy, sx=sx, dy=dy, dx=dx, ch=ch, cw=cw)


def _fill_holes(b):
    """
    Fill interior holes in a binary mask. cv2-only on purpose: this file gets copied
    into the robot project with nothing beyond torch/numpy/cv2 available, so it cannot
    reach for scipy.ndimage.binary_fill_holes the way test.py does.

    Flood the background inward from outside; any background the flood cannot reach is
    enclosed by foreground, i.e. a hole.

    The 1-px pad is load-bearing, not defensive. Seeding the flood at (0, 0) of the
    raw mask assumes that pixel is background — and a vessel touching the frame corner
    makes it foreground, at which point the flood runs through the VESSEL instead and
    the hole survives untouched. Verified against scipy: without the pad, 167 of 300
    random cases disagreed, and a bordering 30x30 square with a 100-px hole came back
    at 800 px instead of 900. Padding guarantees the flood a path all the way around.
    """
    h, w = b.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = b
    inv = (padded == 0).astype(np.uint8)          # 1 = background (outer + holes)
    ffm = np.zeros((h + 4, w + 4), np.uint8)      # floodFill wants a mask 2 px larger
    cv2.floodFill(inv, ffm, (0, 0), 0)            # outer background -> 0
    holes = inv[1:-1, 1:-1].astype(bool)          # still 1 => enclosed => a hole
    return (b.astype(bool) | holes).astype(np.uint8)


def unfit_mask(mask_fit, p):
    """Model-input grid -> crop-frame grid. Exact inverse of fit()."""
    scaled = np.zeros((p["nh"], p["nw"]), dtype=mask_fit.dtype)
    scaled[p["sy"]:p["sy"] + p["ch"], p["sx"]:p["sx"] + p["cw"]] = \
        mask_fit[p["dy"]:p["dy"] + p["ch"], p["dx"]:p["dx"] + p["cw"]]
    if p["scale"] == 1.0:
        return scaled                            # pure offset, no resampling
    return cv2.resize(scaled, (p["W"], p["H"]), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------

class UNetInferencer:
    def __init__(self, ckpt_path, device=None, fit_mode="croppad",
                 target_size=(576, 544), cfg_override=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.cfg = dict(num_classes=1, bilinear=False, amp=True, mean=0.5, std=0.5)
        self.cfg.update(ckpt.get("config", {}) or {})
        if cfg_override:
            self.cfg.update(cfg_override)

        self.n_classes = int(self.cfg["num_classes"])
        self.fit_mode = fit_mode
        self.target_size = tuple(target_size)

        # base_ch comes from the checkpoint's own config. Checkpoints made before it
        # existed are all 64-wide, hence the fallback.
        self.model = UNet(n_channels=1, n_classes=self.n_classes,
                          bilinear=bool(self.cfg["bilinear"]),
                          base_ch=int(self.cfg.get("base_ch", 64)))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device, memory_format=torch.channels_last).eval()

    # -- core -----------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, crop_frame, threshold=0.5, return_prob=False):
        """
        crop_frame : grayscale uint8 (H, W) — ALREADY cropped by depth setting
                     (crop_by_depth), i.e. the frame mm_per_px is calibrated for.

        Returns a mask on the SAME (H, W) grid you passed in, so mm_per_px applies
        directly with no correction factor:
            3-class model -> uint8 ids  {0 bg, 1 vein, 2 artery}
            binary  model -> uint8      {0, 1}   (vessel, type unknown)
        With return_prob=True, returns float32 vessel probability instead (p1+p2 for
        the 3-class model), same grid.
        """
        if crop_frame.ndim == 3:
            crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

        img_fit, _, params = fit(crop_frame, None, self.target_size, self.fit_mode)

        mean, std = float(self.cfg["mean"]), float(self.cfg["std"])
        x = (img_fit.astype(np.float32) / 255.0 - mean) / std
        t = torch.from_numpy(x)[None, None].to(self.device, dtype=torch.float32,
                                               memory_format=torch.channels_last)

        use_amp = bool(self.cfg.get("amp", True)) and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            logits = self.model(t)
        logits = logits.float()

        # Only pay for the normalisation when the caller actually wants probabilities.
        # softmax is strictly monotonic, so argmax(softmax(z)) == argmax(z): computing
        # the softmax and then discarding it via argmax costs an exp over every pixel
        # of every channel for an answer that does not change. Measured on CPU, that
        # was 80 ms of the 191 ms predict() — more than 40% of the deployed latency,
        # spent on arithmetic whose result was thrown away.
        if self.n_classes == 1:
            prob = torch.sigmoid(logits)[0, 0]
            out = prob if return_prob else (prob >= threshold).to(torch.uint8)
        elif return_prob:
            p = F.softmax(logits, dim=1)[0]
            out = p[VEIN] + p[ARTERY]
        else:
            out = logits[0].argmax(0).to(torch.uint8)

        out = out.cpu().numpy()
        if return_prob:
            # unfit_mask is nearest-neighbour; fine for a probability map too, and in
            # croppad mode it is a pure offset anyway.
            return unfit_mask(out.astype(np.float32), params)
        return unfit_mask(out.astype(np.uint8), params)

    # -- geometry -------------------------------------------------------------

    @staticmethod
    def clean(binary, min_area=300):
        """
        Open, close, FILL HOLES, keep the largest connected component.

        The hole fill is not cosmetic — it feeds radius_mm. close(5) only bridges gaps
        up to about its kernel, so a larger hole punched through a predicted lumen
        survived, and every hole subtracts from area_px:

            radius_mm = sqrt(area_px / pi) * mm_per_px

        Measured on Mus-V test with the v3 model: 7.8% of predicted VEINS contain a
        hole, and they under-report the radius by a median of 1.0% and up to 9.1%.
        That is a needle target, so it matters. (The artery is barely affected at 0.6%
        of frames, and the phantom has no holes at all — this is a human-imaging fix.)

        It is close to free, though not because the ground truth is hole-free — 113 of
        7166 ground-truth vessel regions (1.6%) do contain one, so an earlier claim of
        "zero" here was wrong; it had only checked the test splits. The real argument
        is that every one of those holes is annotation noise: the largest in the whole
        project is 68 px (median 11), against a smallest true vein of 232 px, and 106
        of the 113 sit in Mus-V's TRAINING vein masks, which inference never sees. Of
        the 2134 held-out regions, exactly one has a hole. A lumen is anechoic; there
        is no structure inside one for the fill to destroy.
        """
        b = (binary > 0).astype(np.uint8)
        if not b.any():
            return b.astype(bool)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k3)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k5)
        b = _fill_holes(b)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if n <= 1:
            return np.zeros_like(b, dtype=bool)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            return np.zeros_like(b, dtype=bool)
        return lab == i

    def measure(self, mask_crop, mm_per_px, axis_x=None, skin_row=0, min_area=300):
        """
        Geometry in the CROP frame — the coordinate system mm_per_px is defined in.

        mm_per_px is isotropic (the machine renders square pixels: crop_w * mm_per_px
        is the constant 38.3 mm probe aperture at every depth setting), so one scalar
        converts both axes.

            4 cm -> 0.0832    5 cm -> 0.104     (or depth_cm * 10 / crop_h)

        Returns {"vein": {...}, "artery": {...}} — or {"vessel": {...}} for a binary
        model, whose foreground carries no vessel type.
        """
        H, W = mask_crop.shape
        axis_x = W / 2.0 if axis_x is None else axis_x   # probe axis = frame centre

        classes = ((VEIN, "vein"), (ARTERY, "artery")) if self.n_classes > 1 \
            else ((1, "vessel"),)

        out = {}
        for cid, name in classes:
            m = self.clean(mask_crop == cid, min_area)
            if not m.any():
                continue
            M = cv2.moments(m.astype(np.uint8), binaryImage=True)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            ys = np.where(m.any(axis=1))[0]
            top_y, bottom_y = int(ys.min()), int(ys.max())
            area_px = float(m.sum())

            out[name] = {
                "centroid_px":      (cx, cy),           # crop-frame pixels
                "area_px":          area_px,
                "lateral_mm":       (cx - axis_x) * mm_per_px,       # delta, signed
                "depth_mm":         (cy - skin_row) * mm_per_px,     # to centre
                "depth_top_mm":     (top_y - skin_row) * mm_per_px,  # to near wall
                "radius_mm":        float(np.sqrt(area_px / np.pi)) * mm_per_px,
                "diameter_bbox_mm": (bottom_y - top_y + 1) * mm_per_px,
            }
        return out

    @staticmethod
    def mm_per_px_from_depth(depth_cm, crop_h):
        """Fallback when the machine does not hand you a calibration directly."""
        return depth_cm * 10.0 / float(crop_h)

    # -- convenience ----------------------------------------------------------

    def predict_png(self, path, crop=None, **kw):
        """crop = (top, bottom, left, right) pixels to remove, or None."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if crop:
            t, b, l, r = crop
            img = img[t:img.shape[0] - b, l:img.shape[1] - r]
        return self.predict(img, **kw)


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else "0016.png"

    inf = UNetInferencer(ckpt)
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = inf.predict(frame)

    print(f"input {frame.shape} -> mask {mask.shape}  (same grid)  "
          f"ids={np.unique(mask).tolist()}  n_classes={inf.n_classes}")
    for name, g in inf.measure(mask, mm_per_px=0.0832).items():
        print(f"  [{name}] delta={g['lateral_mm']:+.2f} mm  d={g['depth_mm']:.2f} mm  "
              f"r={g['radius_mm']:.2f} mm")
