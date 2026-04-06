#!/usr/bin/env python3
"""
Data collection via A325 video capture card (replaces Clarius API).
Uses OpenCV + pygrabber for device discovery, similar to handler_us.py.
Fixed calibration: 72.35 um/px = 0.07235 mm/px.
"""

import sys
import os
import csv
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot, QTimer
from config import DataInfo

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    print("pygrabber not installed. Run: pip install pygrabber")
    sys.exit(1)

try:
    import comtypes
except ImportError:
    comtypes = None

# ===== Configuration =====
DEVICE_NAME = "A325"

# Clarius original: 557px height, 72.346 um/px → 4.03cm physical height
# A325 cropped:     481px height, same physical area
# → micropixel = 72.34590065828846 * 557 / 481 = 83.776 um/px
MICROPIXEL = 72.34590065828846 * 557 / 481  # ~83.78 um/px (A325 crop calibration)
MM_PER_PX = MICROPIXEL / 1000               # ~0.08378 mm/px

# Defaults matching Clarius settings (for CSV metadata)
DEFAULT_DEPTH = 4           # cm
DEFAULT_GAIN = 50           # %
DEFAULT_FREQUENCY = "8/10.0"  # MHz
DEFAULT_BPP = 4

# Crop parameters (removes iPad Clarius UI borders)
# Raw: 1280x720 -> Crop: 461x481
CROP_TOP = 109
CROP_BOTTOM = 130
CROP_LEFT = 410
CROP_RIGHT = 409


class SavePreviewDialog(QtWidgets.QDialog):
    def __init__(self, image: QtGui.QImage, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Image Preview")
        self.setModal(True)
        self.resize(500, 480)

        layout = QtWidgets.QVBoxLayout(self)

        filename_label = QtWidgets.QLabel(f"Save as: {filename}")
        filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filename_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 8px;")
        layout.addWidget(filename_label)

        img_label = QtWidgets.QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setStyleSheet("border: 1px solid #666; background-color: black;")
        scaled = QtGui.QPixmap.fromImage(image).scaled(
            460, 340, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_label.setPixmap(scaled)
        layout.addWidget(img_label)

        info_label = QtWidgets.QLabel(
            f"Size: {image.width()} x {image.height()}  |  mm/px: {MM_PER_PX}")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(info_label)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        yes_btn = QtWidgets.QPushButton("  Save  ")
        no_btn = QtWidgets.QPushButton("  Cancel  ")
        yes_btn.setDefault(True)
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        yes_btn.clicked.connect(self.accept)
        no_btn.clicked.connect(self.reject)


class MainWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.datainfo = DataInfo(test_name="customer_3d_phantom", test_id="1")

        # Ensure CSV header exists
        if not self.datainfo.meta_file.exists():
            with open(self.datainfo.meta_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "id", "filename", "relative_path",
                    "raw_timestamp", "timestamp",
                    "depth", "gain", "frequency",
                    "bpp", "image_width", "image_height", "micropixel",
                    "test_name", "test_id", "mask_status"])

        with open(self.datainfo.meta_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        self.saved_count = max(0, len(rows) - 1)

        # Crop params
        self.crop_top = CROP_TOP
        self.crop_bottom = CROP_BOTTOM
        self.crop_left = CROP_LEFT
        self.crop_right = CROP_RIGHT

        # Capture state
        self.cap = None
        self.streaming = False
        self.current_frame = None
        self.current_cropped = None
        self.frozen = False

        # Key debounce
        self.key_last_pressed = {Qt.Key.Key_S: 0.0}
        self.key_interval = 1.0

        self._build_ui()

        # Frame timer (~30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab_frame)

    # ================================================================
    #  UI
    # ================================================================
    def _build_ui(self):
        self.setWindowTitle(f"A325 Capture  -  {self.datainfo.test_name}")
        self.setMinimumSize(720, 600)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # ---------- toolbar ----------
        tb = QtWidgets.QHBoxLayout()
        tb.addWidget(QtWidgets.QLabel("Device:"))
        self.device_edit = QtWidgets.QLineEdit(DEVICE_NAME)
        self.device_edit.setFixedWidth(120)
        tb.addWidget(self.device_edit)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.setFixedWidth(90)
        self.btn_connect.clicked.connect(self._toggle_connection)
        tb.addWidget(self.btn_connect)

        tb.addSpacing(16)

        self.btn_freeze = QtWidgets.QPushButton("Freeze")
        self.btn_freeze.setFixedWidth(80)
        self.btn_freeze.setEnabled(False)
        self.btn_freeze.clicked.connect(self._toggle_freeze)
        tb.addWidget(self.btn_freeze)

        self.btn_save = QtWidgets.QPushButton("Save (S)")
        self.btn_save.setFixedWidth(80)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_image)
        tb.addWidget(self.btn_save)

        tb.addStretch()

        self.saved_label = QtWidgets.QLabel(f"Saved: {self.saved_count}")
        self.saved_label.setStyleSheet("font-weight: bold; color: #0a84ff;")
        tb.addWidget(self.saved_label)

        tb.addSpacing(12)
        btn_quit = QtWidgets.QPushButton("Quit")
        btn_quit.setFixedWidth(60)
        btn_quit.clicked.connect(self._shutdown)
        tb.addWidget(btn_quit)

        root.addLayout(tb)

        # ---------- crop controls (collapsible-style row) ----------
        crop_box = QtWidgets.QGroupBox("Crop  (pixels to remove from each edge)")
        crop_layout = QtWidgets.QHBoxLayout(crop_box)
        crop_layout.setContentsMargins(10, 4, 10, 4)
        self.crop_top_spin = self._add_spin("Top", self.crop_top, crop_layout)
        self.crop_bottom_spin = self._add_spin("Bottom", self.crop_bottom, crop_layout)
        self.crop_left_spin = self._add_spin("Left", self.crop_left, crop_layout)
        self.crop_right_spin = self._add_spin("Right", self.crop_right, crop_layout)
        crop_layout.addSpacing(12)
        self.crop_info_label = QtWidgets.QLabel()
        self.crop_info_label.setStyleSheet("color: #888;")
        crop_layout.addWidget(self.crop_info_label)
        crop_layout.addStretch()
        root.addWidget(crop_box)

        # ---------- image display ----------
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        root.addWidget(self.image_label, stretch=1)

        # ---------- help ----------
        help_lbl = QtWidgets.QLabel("Keyboard:  S = Save  |  F = Freeze / Run  |  Q = Quit")
        help_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_lbl.setStyleSheet("color: #666; padding: 2px;")
        root.addWidget(help_lbl)

        self.statusBar().showMessage("Disconnected  -  click Connect to start")

    def _add_spin(self, label: str, value: int, parent_layout):
        parent_layout.addWidget(QtWidgets.QLabel(label + ":"))
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 2000)
        spin.setValue(value)
        spin.setFixedWidth(70)
        spin.valueChanged.connect(self._on_crop_changed)
        parent_layout.addWidget(spin)
        return spin

    def _on_crop_changed(self):
        self.crop_top = self.crop_top_spin.value()
        self.crop_bottom = self.crop_bottom_spin.value()
        self.crop_left = self.crop_left_spin.value()
        self.crop_right = self.crop_right_spin.value()
        self._update_crop_info()

    def _update_crop_info(self):
        """Update crop size and recalculated micropixel display."""
        if self.cap is not None:
            raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raw_w, raw_h = 1280, 720  # fallback
        crop_w = max(1, raw_w - self.crop_left - self.crop_right)
        crop_h = max(1, raw_h - self.crop_top - self.crop_bottom)

        # Recalculate micropixel based on cropped height
        # Clarius reference: 557px = 72.346 um/px at 4cm depth
        ref_physical_um = 72.34590065828846 * 557  # physical height in um
        um_per_px = ref_physical_um / crop_h
        mm_per_px = um_per_px / 1000

        self.crop_info_label.setText(
            f"Crop: {crop_w}x{crop_h}  |  {um_per_px:.2f} um/px  ({mm_per_px:.5f} mm/px)")

    # ================================================================
    #  Connection
    # ================================================================
    def _toggle_connection(self):
        if self.streaming:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        device_name = self.device_edit.text().strip()
        if not device_name:
            self.statusBar().showMessage("Device name is empty")
            return
        try:
            if comtypes is not None:
                try:
                    comtypes.CoInitialize()
                except OSError:
                    pass

            graph = FilterGraph()
            devices = graph.get_input_devices()
            target = next((i for i, n in enumerate(devices) if device_name in n), None)
            if target is None:
                avail = [n for _, n in enumerate(devices)]
                self.statusBar().showMessage(f"'{device_name}' not found.  Available: {avail}")
                return

            self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            fourcc = cv2.VideoWriter.fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2388 / 2)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1668 / 2)

            if not self.cap.isOpened():
                self.statusBar().showMessage("Failed to open capture device")
                return

            # Read actual resolution from device
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            crop_w = actual_w - self.crop_left - self.crop_right
            crop_h = actual_h - self.crop_top - self.crop_bottom

            self.streaming = True
            self.frozen = False
            self.timer.start(33)
            self.btn_connect.setText("Disconnect")
            self.btn_freeze.setEnabled(True)
            self.btn_freeze.setText("Freeze")
            self.btn_save.setEnabled(True)
            self._update_crop_info()
            self.statusBar().showMessage(
                f"Streaming  -  {devices[target]}  |  "
                f"Raw: {actual_w}x{actual_h}  ->  Crop: {crop_w}x{crop_h}")
        except Exception as e:
            self.statusBar().showMessage(f"Connect error: {e}")

    def _disconnect(self):
        self.timer.stop()
        self.streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_connect.setText("Connect")
        self.btn_freeze.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.statusBar().showMessage("Disconnected")

    def _toggle_freeze(self):
        self.frozen = not self.frozen
        if self.frozen:
            self.btn_freeze.setText("Run")
            self.statusBar().showMessage("Frozen  -  press S to save, F to resume")
        else:
            self.btn_freeze.setText("Freeze")
            self.statusBar().showMessage("Streaming")

    # ================================================================
    #  Frame grab & display
    # ================================================================
    def _grab_frame(self):
        if not self.streaming or self.cap is None or self.frozen:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.statusBar().showMessage("Failed to read frame")
            return

        self.current_frame = frame
        cropped = self._crop(frame)
        self.current_cropped = cropped

        # Display: scale to fit the label while keeping aspect ratio
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888).copy()
        label_size = self.image_label.size()
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def _crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        t, b = self.crop_top, self.crop_bottom
        l, r = self.crop_left, self.crop_right
        if t + b >= h:
            t = b = 0
        if l + r >= w:
            l = r = 0
        return img[t:h - b, l:w - r]

    # ================================================================
    #  Save
    # ================================================================
    def _save_image(self):
        if self.current_cropped is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No frame captured yet!")
            return

        gray = cv2.cvtColor(self.current_cropped, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Calculate micropixel from current crop height
        ref_physical_um = 72.34590065828846 * 557
        micropixel = ref_physical_um / h

        idx = f"{self.saved_count:04d}"
        ts = str(int(time.time() * 1000))
        name = f"{idx}_{ts}_T_{self.datainfo.test_id}.png"

        qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
        dialog = SavePreviewDialog(qimg, name, self)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                path = self.datainfo.images_dir / name
                cv2.imwrite(str(path), gray)

                self.saved_count += 1
                with open(self.datainfo.meta_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        idx,
                        name,
                        str(self.datainfo.images_dir / name),
                        "",                 # raw_timestamp
                        ts,
                        DEFAULT_DEPTH,      # depth (4 cm)
                        DEFAULT_GAIN,       # gain (50%)
                        DEFAULT_FREQUENCY,  # frequency (8/10.0 MHz)
                        DEFAULT_BPP,        # bpp (4)
                        w,
                        h,
                        micropixel,         # um/px (recalculated from crop height)
                        self.datainfo.test_name,
                        self.datainfo.test_id,
                        "ND"
                    ])

                self.saved_label.setText(f"Saved: {self.saved_count}")
                self.statusBar().showMessage(f"Saved: {name}  ({w}x{h})")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {e}")
        else:
            self.statusBar().showMessage("Save cancelled")

    # ================================================================
    #  Keyboard
    # ================================================================
    def keyPressEvent(self, event):
        key = event.key()
        now = time.time()

        if key == Qt.Key.Key_S:
            if now - self.key_last_pressed.get(Qt.Key.Key_S, 0) < self.key_interval:
                return
            self.key_last_pressed[Qt.Key.Key_S] = now
            self._save_image()
        elif key == Qt.Key.Key_F:
            self._toggle_freeze()
        elif key == Qt.Key.Key_Q:
            self._shutdown()

    @Slot()
    def _shutdown(self):
        self._disconnect()
        QtWidgets.QApplication.quit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWidget()
    w.resize(900, 680)
    w.show()
    sys.exit(app.exec())
