# This handler is used to read out the ultrasound frame from the video capture
# When we get the frame and then pipeline it through the model for inference (if the inference speed is slower than the frame rate, we need to abort some raw frames)
# post process the segmentation results with the rotation of the US probe from the handler vision dataclass. then we could get the thickness of the vein and muscle (by the pixel estimate)
# Store the results into related dataclass
# It should has ability to visualize the realtime segmentation results

import cv2
import numpy as np
import threading
import time
import os
import sys
from collections import deque
from pygrabber.dshow_graph import FilterGraph
from tools.ml.inferencer import UNetInferencer
from config_loader import cfg
from dataclasses import dataclass, field, asdict
try:
    import comtypes  # Needed for explicit COM initialization on worker threads
except ImportError:
    comtypes = None

# ===== Dataclasses =====
@dataclass(frozen=True)
class VesselMetricsSnapshot:
    '''
    cx: 稳健过滤后的血管掩膜重心 x 像素坐标（用于控制/显示）
    cy: 稳健过滤后的血管掩膜重心 y 像素坐标
    raw_cx: 未经过稳健/MAD/σ 过滤的原始重心 x
    raw_cy: 未经过稳健/MAD/σ 过滤的原始重心 y
    top_y: 血管二值掩膜外接包围盒顶部 y 像素行号
    bottom_y: 包围盒底部 y 像素行号
    left_x: 包围盒左侧 x 像素列号
    right_x: 包围盒右侧 x 像素列号
    height_px: 包围盒在 y 方向像素高度 (bottom_y - top_y + 1)
    width_px: 包围盒在 x 方向像素宽度 (right_x - left_x + 1)
    height_mm: 等效圆直径 (mm) = 2 × √(active_pixels / π) × mm_per_px（比包围盒更鲁棒）
    width_mm: 由 width_px 按 mm_per_px 换算的包围盒"宽度" (mm)
    depth_mm: 血管重心深度 (mm) = rcy × mm_per_px（从图像顶部/皮肤表面到血管中心）
    offset_x_px: 稳健重心相对图像水平中心 (w/2) 的水平偏移像素值 (右正左负)
    offset_x_mm: 上述偏移换算为毫米
    active_pixels: 当前掩膜中前景 (True / >127) 像素总数（面积估计基础）
    mm_per_px: 当前帧估算的毫米/像素比例（纵向基准，用于所有 mm 换算）
    timestamp: 该组指标生成的时间戳（time.time() 秒）
    '''
    cx: float; cy: float; raw_cx: float; raw_cy: float
    top_y: int; bottom_y: int; left_x: int; right_x: int
    height_px: int; width_px: int
    height_mm: float; width_mm: float; depth_mm: float
    offset_x_px: float; offset_x_mm: float
    active_pixels: int; mm_per_px: float
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

class VideoRecorder:
    def __init__(self, device_name="A325"):
        self.device_name = device_name

    def init_capture_device(self) -> cv2.VideoCapture:
        # get the video capture device
        # IMPORTANT (Windows / COM): When using DirectShow (pygrabber) from a background
        # thread, that thread must call CoInitialize/CoInitializeEx before creating
        # COM objects. The main thread is usually initialized implicitly, but worker
        # threads are not. Failing to do so raises WinError -2147221008.
        if comtypes is not None:
            try:
                # If already initialized this may raise, so we suppress OSError
                comtypes.CoInitialize()
            except OSError:
                # Already initialized on this thread; safe to ignore
                pass
            except Exception as e:
                print(f"[WARN] COM initialization failed (continuing): {e}")

        graph = FilterGraph()
        devices = graph.get_input_devices()
        target = next((i for i,n in enumerate(devices) if self.device_name in n), None)
        if target is None:
            raise RuntimeError(f"Failed to find device containing '{self.device_name}'. Please check the connection.")

        print(f"Using device: {devices[target]}")

        # Initialize the video capture
        cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter.fourcc(*'MJPG')  # type: ignore[attr-defined]
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        desired_w, desired_h = 2388/2, 1668/2 # This is for iPad
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  desired_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open device {target}.")
        return cap
    
class USHandler():
    def __init__(self, recorder: VideoRecorder, inferencer: UNetInferencer, enable_visualization: bool):

        self.recorder = recorder
        self.cap = None
        self.running = False
        _us = cfg.section("ultrasound")
        self.crop_params = {
            'crop_top': _us.get("crop_top", 108),
            'crop_bottom': _us.get("crop_bottom", 130),
            'crop_left': _us.get("crop_left", 414),
            'crop_right': _us.get("crop_right", 411),
        }

        # handler
        self.recorder = recorder
        self.inferencer = inferencer

        # status
        self.running = False
        self.enable_inference = False
        self.enable_visualization = enable_visualization

        # transparency for overlay
        self.alpha = _us.get("overlay_alpha", 0.3)

        # 视频录制
        self._video_writer: cv2.VideoWriter | None = None
        self._video_recording: bool = False

        # === 历史数据 (使用 dataclass) ===
        self.history_window_sec = _us.get("robust_window_sec", 1.0)
        self._metrics_history = deque(maxlen=1000)   # VesselMetricsSnapshot
        self._last_metrics: VesselMetricsSnapshot | None = None
        self._hist_lock = threading.Lock()

    def start_thread(self):
        try:
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start USHandler thread: {e}")

    def run(self):
        try:
            self.cap = self.recorder.init_capture_device()
        except Exception as e:
            print(f"[US] Failed to open capture device: {e}")
            return
        self.running = True
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[US] Failed to read frame, stopping.")
                    break

                # Apply cropping to map the size for the model
                cropped_frame = self._crop_image(frame, **self.crop_params)

                if self.enable_inference:
                    # Execute inference and overlay mask
                    processed_frame = self.process_inference(cropped_frame, enable_visualization=self.enable_visualization)
                else:
                    # Directly use the original frame
                    processed_frame = cropped_frame

                if self.enable_visualization:
                    # key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        # start and stop the inference
                        print("Toggling inference...")
                        self.enable_inference = not self.enable_inference
                    elif key == ord('p'):
                        # print the latest metrics
                        latest_metrics = self.get_latest_vessel_metrics_snapshot()
                        if latest_metrics:
                            print("Latest Metrics:")
                            for k, v in latest_metrics.to_dict().items():
                                print(f"  {k}: {v}")
                        else:
                            print("No metrics available.")
                    elif key == ord('q'):
                        # quit the application
                        self.stop()
                    # Show the processed frame
                    # US 录制（保存未 crop 的原始帧，原始分辨率）
                    if self._video_recording:
                        if self._video_writer is None:
                            _rh, _rw = frame.shape[:2]
                            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                            self._video_writer = cv2.VideoWriter(self._video_path, fourcc, 30.0, (_rw, _rh))
                        if self._video_writer is not None:
                            self._video_writer.write(frame)

                    _h, _w = processed_frame.shape[:2]
                    _scale = 480.0 / _w
                    _us_small = cv2.resize(processed_frame, (480, int(_h * _scale)), interpolation=cv2.INTER_AREA)
                    if self._video_recording:
                        cv2.circle(_us_small, (465, 15), 8, (0, 0, 255), -1)
                    cv2.imshow("Processed Frame", _us_small)
                    if not hasattr(self, '_win_positioned'):
                        cv2.moveWindow("Processed Frame", 1440, 480)
                        self._win_positioned = True
                    cv2.waitKey(1)

                # sleep
                time.sleep(0.033)  # ~30 FPS
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def stop(self):
        self.running = False
        self.stop_video_capture()

    def start_video_capture(self, timestamp: str):
        """开始 US 视频录制，timestamp 与 vision 对齐。writer 延迟到第一帧创建。"""
        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "trials", "videos", "us")
        os.makedirs(_dir, exist_ok=True)
        self._video_path = os.path.join(_dir, f"us_{timestamp}.mp4")
        self._video_writer = None  # 延迟创建
        self._video_recording = True
        print(f"[US] Video capture started: us_{timestamp}.mp4")

    def stop_video_capture(self):
        """停止 US 视频录制。"""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._video_recording = False

    def set_inference_enabled(self, enabled: bool):
        """Set whether to enable inference"""
        self.enable_inference = enabled

    def process_inference(self, frame, enable_visualization=False):
        """Process frame with UNet inference and overlay mask"""
        try:
            # Always have an output reference; clone only if we will draw overlays
            inference_display = frame.copy() if enable_visualization else frame
            # Convert to grayscale for inference
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # To np array
            img_np = np.array(frame_gray)
            
            # Run inference (threshold hot-reloadable)
            inference_result = self.inferencer.predict_array(
                img_np, threshold=cfg.get("ultrasound.inference_threshold", 0.75)
            )

            # Ensure mask and image dimensions match (inferencer 输出 576×544，需 resize 回原图)
            if inference_result.shape[:2] != frame.shape[:2]:
                inference_result = cv2.resize(
                    inference_result, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # === Keep only largest connected component ===
            try:
                mask_tmp = inference_result
                if mask_tmp.ndim == 3:  # 如果是多通道取第一通道
                    mask_tmp = mask_tmp[...,0]
                if mask_tmp.dtype != np.uint8:
                    mask_tmp = mask_tmp.astype(np.uint8)
                bin_mask = (mask_tmp > 127).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                if num_labels > 1:
                    # stats[0] is background, the rest are foreground
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    max_label = 1 + int(np.argmax(areas))
                    largest = (labels == max_label).astype(np.uint8)
                    inference_result = (largest * 255).astype(np.uint8)
                else:
                    # No foreground detected
                    inference_result = np.zeros_like(bin_mask, dtype=np.uint8)
            except Exception as e:
                print(f"Largest component filtering error: {e}")

            # Compute connected components and measurements
            mask_active = inference_result > 127
            active_pixels = int(np.sum(mask_active))


            if active_pixels > 0:
                ys, xs = np.where(mask_active)
                top_y, bottom_y = int(ys.min()), int(ys.max())
                left_x, right_x = int(xs.min()), int(xs.max())
                height_px = bottom_y - top_y + 1
                width_px = right_x - left_x + 1
                # 重心（与 predict.ipynb cv2.moments 数学等价）
                cx = float(xs.mean())
                cy = float(ys.mean())
                # 稳健重心：MAD + 2σ 滤波
                rcx, rcy = self._robust_centroid_internal(raw_candidate=(cx, cy))
                h_frame, w_frame = frame.shape[:2]
                mm_per_px = cfg.get("ultrasound.mm_per_px", 0.07235)
                # 等效圆半径（与 predict.ipynb 一致，比包围盒更鲁棒）
                _radius_px = float(np.sqrt(active_pixels / np.pi))
                _diameter_px = _radius_px * 2.0
                height_mm = _diameter_px * mm_per_px   # 等效直径 (mm)
                width_mm = width_px * mm_per_px         # 包围盒宽度 (mm)
                # 深度：用重心 cy（与 predict.ipynb 一致，代表血管中心深度）
                depth_mm = rcy * mm_per_px
                mid_x = w_frame / 2.0
                offset_x_px = rcx - mid_x
                offset_x_mm = offset_x_px * mm_per_px
                # filter out small detections
                if height_mm < cfg.get("ultrasound.min_detection_height_mm", 1.0):
                    with self._hist_lock:
                        self._last_metrics = None
                    return frame
                _snap = VesselMetricsSnapshot(
                    cx=rcx, cy=rcy, raw_cx=cx, raw_cy=cy,
                    top_y=top_y, bottom_y=bottom_y, left_x=left_x, right_x=right_x,
                    height_px=height_px, width_px=width_px,
                    height_mm=height_mm, width_mm=width_mm, depth_mm=depth_mm,
                    offset_x_px=offset_x_px, offset_x_mm=offset_x_mm,
                    active_pixels=active_pixels, mm_per_px=mm_per_px,
                    timestamp=time.time()
                )
                # Store to latest + history
                with self._hist_lock:
                    self._last_metrics = _snap
                    self._metrics_history.append(_snap)

                if enable_visualization:
                    inference_display[mask_active, 0] = inference_display[mask_active, 0] * (1 - self.alpha)
                    inference_display[mask_active, 1] = inference_display[mask_active, 1] * (1 - self.alpha) + 255 * self.alpha
                    inference_display[mask_active, 2] = inference_display[mask_active, 2] * (1 - self.alpha)
                    self._draw_overlay(inference_display, _snap.to_dict())
            else:
                with self._hist_lock:
                    self._last_metrics = None
            return inference_display
            
        except Exception as e:
            print(f"Inference processing error: {e}")
            return frame

    def _robust_centroid_internal(self, raw_candidate: tuple[float,float] | None = None):
        return self._get_robust_centroid(self.history_window_sec, raw_candidate=raw_candidate)

    def _get_robust_centroid(self, window_sec: float | None = None, raw_candidate: tuple[float,float] | None = None, k_std: float = 2.0) -> tuple[float, float]:
        """返回窗口内稳健重心 (cx, cy)。

        步骤:
          1. 取最近 window_sec 秒(默认 self.history_window_sec) 的所有检测点 (cx, cy)。
          2. 若数量 <5 直接返回均值 (或 raw_candidate 兜底)。
          3. 对 x 与 y 分别做一次 MAD 过滤 (|z_MAD| < 2.5)。
          4. 在剩余点上计算均值和标准差, 再执行 2σ(可调 k_std) 过滤 (任一轴超范围即剔除)。
          5. 若过滤后剩余点>=3 取其均值, 否则回退到 MAD 过滤结果或原始均值。

        参数:
          window_sec: 时间窗口秒; None 使用默认。
          raw_candidate: 在无历史时返回的候选(raw cx, raw cy)。
          k_std: σ 过滤阈值 (默认 2.0)。
        """
        win = float(window_sec) if window_sec is not None else float(self.history_window_sec)
        now = time.time()
        with self._hist_lock:
            pts = [m for m in self._metrics_history if (now - m.timestamp) <= win]
        if not pts:
            if raw_candidate is not None:
                return float(raw_candidate[0]), float(raw_candidate[1])
            return 0.0, 0.0

        # 使用存储的原始重心 raw_cx/raw_cy 进行稳健统计，避免二次平滑
        xs_all = np.array([p.raw_cx for p in pts], dtype=float)
        ys_all = np.array([p.raw_cy for p in pts], dtype=float)

        if xs_all.size < 5:
            return float(xs_all.mean()), float(ys_all.mean())

        def _mad_filter(arr: np.ndarray, thresh: float = 2.5):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            if mad < 1e-9:
                return np.ones_like(arr, dtype=bool)
            z = np.abs(arr - med) / (1.4826 * mad)
            return z < thresh

        keep_x = _mad_filter(xs_all)
        keep_y = _mad_filter(ys_all)
        keep_mad = keep_x & keep_y
        xs_mad = xs_all[keep_mad]
        ys_mad = ys_all[keep_mad]
        if xs_mad.size < 3:  # MAD 太严，退回全部
            xs_mad, ys_mad = xs_all, ys_all

        # 2σ 过滤 (任一轴超出均值±k_std*std 即剔除)
        mx, sx = float(xs_mad.mean()), float(xs_mad.std(ddof=0))
        my, sy = float(ys_mad.mean()), float(ys_mad.std(ddof=0))
        if sx < 1e-9: sx = 0.0
        if sy < 1e-9: sy = 0.0
        if sx == 0.0 and sy == 0.0:
            return mx, my
        mask_sigma = np.ones(xs_mad.shape, dtype=bool)
        if sx > 0:
            mask_sigma &= (np.abs(xs_mad - mx) <= k_std * sx)
        if sy > 0:
            mask_sigma &= (np.abs(ys_mad - my) <= k_std * sy)
        xs_final = xs_mad[mask_sigma]
        ys_final = ys_mad[mask_sigma]
        if xs_final.size >= 3:
            return float(xs_final.mean()), float(ys_final.mean())
        # 回退: 使用 MAD 过滤均值
        return mx, my

    def _draw_overlay(self, img, m):
        try:
            cx, cy = int(round(m['cx'])), int(round(m['cy']))
            color_box = (0, 255, 0)
            cv2.rectangle(img, (m['left_x'], m['top_y']), (m['right_x'], m['bottom_y']), color_box, 2)
            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
            h, w = img.shape[:2]
            mid_x = int(w / 2)
            cv2.line(img, (mid_x, 0), (mid_x, h), (0, 0, 255), 2)
            cv2.line(img, (mid_x, cy), (cx, cy), (255, 0, 0), 2)
            lines = [
                f"depth:{m['depth_mm']:.1f}mm",
                f"H:{m['height_mm']:.1f}mm W:{m['width_mm']:.1f}mm",
                f"offX:{m['offset_x_mm']:+.1f}mm",
                f"pix/mm:{1.0/m['mm_per_px']:.2f}" if m['mm_per_px']>0 else "",
            ]
            y0 = 25
            for i, text in enumerate(lines):
                if not text:
                    continue
                cv2.putText(img, text, (10, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    def _crop_image(self, img: np.ndarray, crop_top: int = 0, crop_bottom: int = 0, 
                   crop_left: int = 0, crop_right: int = 0) -> np.ndarray:
        """Crop the given image by removing specified pixels from each border."""
        h, w = img.shape[:2]
        
        # Ensure cropping isn't too large
        if crop_top + crop_bottom >= h:
            crop_top = crop_bottom = 0
        if crop_left + crop_right >= w:
            crop_left = crop_right = 0
            
        return img[crop_top:h - crop_bottom, crop_left:w - crop_right]

    # ===== 对外 API =====
    def get_latest_vessel_metrics_snapshot(self) -> VesselMetricsSnapshot | None:
        """返回最近的 VesselMetricsSnapshot 实例或 None"""
        with self._hist_lock:
            return self._last_metrics


def init_us_handler(enable_visualization=True):
    recorder = VideoRecorder(device_name="A325")

    # 先检查设备是否存在（不创建 capture，只检查名称）
    try:
        if comtypes is not None:
            try:
                comtypes.CoInitialize()
            except OSError:
                pass
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        target = next((i for i, n in enumerate(devices) if recorder.device_name in n), None)
        if target is None:
            raise RuntimeError(f"Device '{recorder.device_name}' not found. Available: {[n for _, n in enumerate(devices)]}")
        print(f"[US] Found device: {devices[target]}")
    except Exception as e:
        print(f"[US] Device not available: {e}")
        return None

    # 设备存在，再加载模型
    try:
        # _ckpt = cfg.get("ultrasound.model_checkpoint", "tools/ml/checkpoints/checkpoint_expC_mixed_71_dice_0.9257_20260313_093551.pth")
        _ckpt = cfg.get("ultrasound.model_checkpoint", "tools/ml/checkpoints/best_model_v3.pth")
        inferencer = UNetInferencer(ckpt_path=_ckpt)
        print(f"[US] UNet inferencer initialized: {_ckpt}")
    except Exception as e:
        print(f"[US] UNet inferencer initialization failed: {e}")
        return None

    us_handler = USHandler(recorder, inferencer, enable_visualization=enable_visualization)
    return us_handler

if __name__ == "__main__":

    us_handler = init_us_handler(enable_visualization=True)
    # start the handler
    if us_handler:
        us_handler.set_inference_enabled(True)
        us_handler.start_thread()
