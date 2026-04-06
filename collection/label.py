#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6-based circular brush annotation tool for annotating phantom images.
Features:
- Free drawing with circular brush
- Adjustable brush size, multiple brush types
- Real-time colored overlay display
- Save grayscale mask images (pixel values per brush type)
- Numpy-backed mask for fast export (no pixel-by-pixel loop)

Usage:
    python dataLabel.py                          # default: phantom_1
    python dataLabel.py phantom_2      # specify test_name
    python dataLabel.py phantom_2 1    # specify test_name and test_id
"""

import sys
import os
import csv
import math
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSlider, QSpinBox, QTextEdit,
    QComboBox
)

from config import DataInfo

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
MIN_BRUSH_SIZE = 5
MAX_BRUSH_SIZE = 100
DEFAULT_BRUSH_SIZE = 20

STATUS_ND = "ND"
STATUS_TRUE = "true"
STATUS_TEST = "test"
STATUS_PASS = "pass"

BRUSH_TYPES = {
    "vein": {
        "color": QColor(0, 255, 0, 50),
        "pixel_value": 255,
        "display_name": "Vein"
    },
    "tbd1": {
        "color": QColor(0, 0, 255, 50),
        "pixel_value": 128,
        "display_name": "tbd1"
    },
    "tbd2": {
        "color": QColor(255, 0, 0, 50),
        "pixel_value": 64,
        "display_name": "tbd2"
    }
}
# ────────────────────────────────────────────────────────────────────────────────


class PaintView(QGraphicsView):
    """Canvas with circular brush drawing backed by a numpy mask array."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        self.paint_pixmap = None
        self.paint_item = QGraphicsPixmapItem()
        self.scene().addItem(self.paint_item)

        # Numpy mask: same size as image, stores pixel_value per brush type
        self._mask_np = None

        self.brush_size = DEFAULT_BRUSH_SIZE
        self.current_brush_type = list(BRUSH_TYPES.keys())[0]
        self.is_painting = False
        self.last_pt = QPointF()

        self.setMouseTracking(True)

    def load_image(self, path: Path):
        pix = QPixmap(str(path))
        self.pixmap_item.setPixmap(pix)
        self.setSceneRect(QRectF(pix.rect()))

        self.paint_pixmap = QPixmap(pix.size())
        self.paint_pixmap.fill(Qt.GlobalColor.transparent)
        self.paint_item.setPixmap(self.paint_pixmap)

        self._mask_np = np.zeros((pix.height(), pix.width()), dtype=np.uint8)

    def set_brush_size(self, s: int):
        self.brush_size = s

    def set_brush_type(self, brush_type: str):
        if brush_type in BRUSH_TYPES:
            self.current_brush_type = brush_type

    def get_current_brush_color(self):
        return BRUSH_TYPES[self.current_brush_type]["color"]

    # ── Mouse events ──────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.paint_pixmap:
            self.is_painting = True
            self.last_pt = self.mapToScene(ev.position().toPoint())
            self._stamp(self.last_pt)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pt = self.mapToScene(ev.position().toPoint())
        if self.is_painting:
            self._stroke(self.last_pt, pt)
            self.last_pt = pt
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.is_painting = False
        super().mouseReleaseEvent(ev)

    # ── Drawing helpers ───────────────────────────────────────────
    def _stamp(self, center: QPointF):
        """Draw one circle at center, update both visual overlay and numpy mask."""
        r = self.brush_size / 2.0
        cx, cy = center.x(), center.y()

        # Visual overlay
        painter = QPainter(self.paint_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setBrush(QBrush(self.get_current_brush_color()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, r, r)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)

        # Numpy mask (no anti-aliasing, hard circle)
        self._stamp_mask(cx, cy, r)

    def _stamp_mask(self, cx: float, cy: float, r: float):
        """Burn a filled circle into self._mask_np."""
        if self._mask_np is None:
            return
        h, w = self._mask_np.shape
        pv = BRUSH_TYPES[self.current_brush_type]["pixel_value"]

        # Bounding box (clipped)
        y0 = max(0, int(cy - r))
        y1 = min(h, int(cy + r) + 1)
        x0 = max(0, int(cx - r))
        x1 = min(w, int(cx + r) + 1)
        if y0 >= y1 or x0 >= x1:
            return

        ys = np.arange(y0, y1, dtype=np.float32)
        xs = np.arange(x0, x1, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inside = dist2 <= r * r
        self._mask_np[y0:y1, x0:x1][inside] = pv

    def _stroke(self, p0: QPointF, p1: QPointF):
        """Interpolate circles along a line between two points."""
        dist = math.hypot(p1.x() - p0.x(), p1.y() - p0.y())
        step = max(1.0, self.brush_size / 4.0)
        steps = max(1, int(dist / step))

        # Batch: collect all centers, draw visual in one painter session
        centers = []
        for i in range(steps + 1):
            t = i / steps
            x = p0.x() + (p1.x() - p0.x()) * t
            y = p0.y() + (p1.y() - p0.y()) * t
            centers.append((x, y))

        r = self.brush_size / 2.0
        color = self.get_current_brush_color()

        # One painter for all circles in this stroke segment
        painter = QPainter(self.paint_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        for cx, cy in centers:
            painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)

        # Numpy mask
        for cx, cy in centers:
            self._stamp_mask(cx, cy, r)

    def clear_painting(self):
        if self.paint_pixmap:
            self.paint_pixmap.fill(Qt.GlobalColor.transparent)
            self.paint_item.setPixmap(self.paint_pixmap)
        if self._mask_np is not None:
            self._mask_np.fill(0)

    def get_mask_image(self) -> QImage | None:
        """Return grayscale QImage from numpy mask (instant, no pixel loop)."""
        if self._mask_np is None:
            return None
        h, w = self._mask_np.shape
        # QImage from numpy: must keep reference alive
        data = self._mask_np.tobytes()
        img = QImage(data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        return img


class MainWindow(QMainWindow):
    def __init__(self, test_name="phantom_1", test_id="1"):
        super().__init__()
        self.dataInfo = DataInfo(test_name=test_name, test_id=test_id)
        self.dataInfo.ensure_dirs()
        self.setWindowTitle(f"Annotation Tool: {self.dataInfo.test_name} (Test {self.dataInfo.test_id})")

        self._setup_font()

        # Load CSV
        with open(self.dataInfo.meta_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            self.rows = list(reader)

        # Find status column
        candidates = [c for c in self.fieldnames
                      if c.lower() in
                         ('status', 'mask_status', 'label_status',
                          'annotation_status', 'annot_status')]
        if not candidates:
            QMessageBox.critical(self, "Error",
                "Cannot find a status column in CSV. "
                "Add a column named mask_status.")
            sys.exit(1)
        self.status_col = candidates[0]

        # Find path column
        if 'relative_path' in self.fieldnames:
            self.path_col = 'relative_path'
        elif 'filename' in self.fieldnames:
            self.path_col = 'filename'
        else:
            QMessageBox.critical(self, "Error",
                "Cannot find image path column in CSV. "
                "Add 'relative_path' or 'filename'.")
            sys.exit(1)

        # Build pending list (ND status only)
        self.pending = [i for i, r in enumerate(self.rows)
                        if r[self.status_col] == STATUS_ND]
        if not self.pending:
            QMessageBox.information(self, "Done", "No images with ND status to process.")
            sys.exit(0)
        self.cur = 0

        self._build_ui()
        self.load_current()

    # ================================================================
    #  UI
    # ================================================================
    def _setup_font(self):
        self._font = QFont("SimSun", 10)
        QApplication.instance().setFont(self._font)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(8, 8, 8, 4)

        # Canvas
        self.view = PaintView()
        vlay.addWidget(self.view, stretch=1)

        # Brush type
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Brush:"))
        self.brush_type_combo = QComboBox()
        for btype, binfo in BRUSH_TYPES.items():
            self.brush_type_combo.addItem(binfo["display_name"], btype)
        self.brush_type_combo.currentIndexChanged.connect(self._on_brush_type)
        row1.addWidget(self.brush_type_combo)

        row1.addSpacing(20)
        row1.addWidget(QLabel("Size:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.brush_slider.setValue(DEFAULT_BRUSH_SIZE)
        row1.addWidget(self.brush_slider)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.brush_spin.setValue(DEFAULT_BRUSH_SIZE)
        row1.addWidget(self.brush_spin)

        self.brush_slider.valueChanged.connect(self.brush_spin.setValue)
        self.brush_spin.valueChanged.connect(self.brush_slider.setValue)
        self.brush_slider.valueChanged.connect(self.view.set_brush_size)

        vlay.addLayout(row1)

        # Buttons
        row2 = QHBoxLayout()
        self.btn_prev  = QPushButton("Previous")
        self.btn_next  = QPushButton("Next")
        self.btn_clear = QPushButton("Clear")
        self.btn_pass  = QPushButton("Skip (pass)")
        self.btn_test  = QPushButton("Save as Test")
        self.btn_true  = QPushButton("Save as True")
        for btn in (self.btn_prev, self.btn_next, self.btn_clear,
                    self.btn_pass, self.btn_test, self.btn_true):
            row2.addWidget(btn)
        vlay.addLayout(row2)

        self.btn_prev.clicked.connect(self.goto_prev)
        self.btn_next.clicked.connect(self.goto_next)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_pass.clicked.connect(self._mark_pass)
        self.btn_test.clicked.connect(lambda: self._save(STATUS_TEST))
        self.btn_true.clicked.connect(lambda: self._save(STATUS_TRUE))

        # Status + log
        self.status_label = QLabel()
        vlay.addWidget(self.status_label)

        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(100)
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(
            "QTextEdit { background: #f8f8f8; border: 1px solid #ccc; padding: 4px; }")
        vlay.addWidget(self.log_area)

        help_lbl = QLabel(
            "Left-click drag to draw  |  Clear: erase all  |  "
            "Save as Test/True: save mask + advance  |  Skip: mark pass + advance")
        help_lbl.setStyleSheet("color: #666; padding: 2px;")
        help_lbl.setWordWrap(True)
        vlay.addWidget(help_lbl)

        self.log("Annotation tool started")

    def _on_brush_type(self, index):
        btype = self.brush_type_combo.itemData(index)
        if btype and btype in BRUSH_TYPES:
            self.view.set_brush_type(btype)
            self.log(f"Brush: {BRUSH_TYPES[btype]['display_name']} "
                     f"(pixel={BRUSH_TYPES[btype]['pixel_value']})")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{ts}] {msg}")
        c = self.log_area.textCursor()
        c.movePosition(c.MoveOperation.End)
        self.log_area.setTextCursor(c)

    # ================================================================
    #  Navigation
    # ================================================================
    def load_current(self):
        idx = self.pending[self.cur]
        row = self.rows[idx]
        p = Path(row[self.path_col])
        if not p.exists():
            p = self.dataInfo.images_dir / Path(row[self.path_col]).name
        if not p.exists():
            QMessageBox.critical(self, "Error", f"Image not found:\n{row[self.path_col]}")
            sys.exit(1)

        self.current_path = p
        self.view.load_image(p)
        self.status_label.setText(
            f"[{self.cur + 1}/{len(self.pending)}]  {p.name}  —  {row[self.status_col]}")
        self.log(f"Loaded: {p.name}")

        self.btn_prev.setEnabled(self.cur > 0)
        self.btn_next.setEnabled(self.cur < len(self.pending) - 1)

    def goto_prev(self):
        if self.cur > 0:
            self.cur -= 1
            self.load_current()

    def goto_next(self):
        if self.cur < len(self.pending) - 1:
            self.cur += 1
            self.load_current()

    # ================================================================
    #  Actions
    # ================================================================
    def _clear(self):
        self.view.clear_painting()
        self.log("Cleared")

    def _mark_pass(self):
        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = STATUS_PASS
        self._write_csv()
        self.log(f"Skipped: {self.current_path.name}")
        self._advance()

    def _save(self, status: str):
        if not self.view.pixmap_item.pixmap():
            self.log("Error: no image loaded")
            return

        mask_img = self.view.get_mask_image()
        if mask_img is None:
            # Empty mask (nothing drawn)
            pix = self.view.pixmap_item.pixmap()
            mask_img = QImage(pix.size(), QImage.Format.Format_Grayscale8)
            mask_img.fill(0)

        base = self.current_path.stem
        out_name = f"{base}_mask.png"
        out_path = self.dataInfo.masks_dir / out_name

        if not mask_img.save(str(out_path)):
            self.log(f"FAILED to save mask: {out_path}")
            return

        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = status

        if 'mask_path' not in self.fieldnames:
            self.fieldnames.append('mask_path')
        self.rows[idx]['mask_path'] = str(out_path)

        self._write_csv()
        self.log(f"Saved ({status}): {out_name}")
        self._advance()

    def _advance(self):
        self.pending.pop(self.cur)
        if not self.pending:
            self.log("All done!")
            QMessageBox.information(self, "Done", "All images annotated.")
            QApplication.quit()
            return
        if self.cur >= len(self.pending):
            self.cur = len(self.pending) - 1
        self.log(f"Remaining: {len(self.pending)}")
        self.load_current()

    def _write_csv(self):
        with open(self.dataInfo.meta_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation tool")
    parser.add_argument("test_name", nargs="?", default="phantom_1",
                        help="Dataset name (e.g. phantom_1, phantom_2)")
    parser.add_argument("test_id", nargs="?", default="1", help="Test ID")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(test_name=args.test_name, test_id=args.test_id)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())
