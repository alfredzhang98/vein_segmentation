#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6-based circular brush annotation tool for annotating phantom images.
Features:
- Free drawing with circular brush
- Adjustable brush size
- Real-time red display of drawing area
- Save binary mask images
- Fixed color overlay issues
- Fixed mask color issues and simplified operation flow
"""

import sys
import csv
import math
import time
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QPolygonF, QFont, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSlider, QSpinBox, QTextEdit,
    QComboBox
)

from DataInfo import DataInfo
# ────────────────────────────────────────────────────────────────────────────────
# Configuration parameters
MIN_BRUSH_SIZE = 5   # Minimum brush size
MAX_BRUSH_SIZE = 100 # Maximum brush size
DEFAULT_BRUSH_SIZE = 20  # Default brush size

# Status constants
STATUS_ND = "ND"              # To be processed
STATUS_TRUE = "true"          # True sample (with annotation content)
STATUS_TEST = "test"          # Test sample (blank annotation or for testing)
STATUS_PASS = "pass"          # Skip

# Brush types with colors and pixel values
BRUSH_TYPES = {
    "vein": {
        "color": QColor(0, 255, 0, 50),  # Green semi-transparent
        "pixel_value": 255,
        "display_name": "Vein"
    },
    "tbd1": {
        "color": QColor(0, 0, 255, 50),  # Blue semi-transparent
        "pixel_value": 128,
        "display_name": "tbd1"
    },
    "tbd2": {
        "color": QColor(255, 0, 0, 50),  # Red semi-transparent
        "pixel_value": 64,
        "display_name": "tbd2"
    }
}
# ────────────────────────────────────────────────────────────────────────────────

class PaintView(QGraphicsView):
    """Use Source to directly draw circular brush, anti-aliasing + no color overlay."""
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))

        # Base image
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        # Drawing layer
        self.paint_pixmap = None
        self.paint_item = QGraphicsPixmapItem()
        self.scene().addItem(self.paint_item)

        self.brush_size = DEFAULT_BRUSH_SIZE
        # BRUSH_TYPES first item is default
        self.current_brush_type = list(BRUSH_TYPES.keys())[0]
        self.is_painting = False
        self.last_pt = QPointF()

        self.setMouseTracking(True)

    def load_image(self, path: Path):
        pix = QPixmap(str(path))
        self.pixmap_item.setPixmap(pix)
        self.setSceneRect(pix.rect())

        # Initialize a transparent canvas
        self.paint_pixmap = QPixmap(pix.size())
        self.paint_pixmap.fill(Qt.GlobalColor.transparent)
        self.paint_item.setPixmap(self.paint_pixmap)

    def set_brush_size(self, s: int):
        self.brush_size = s

    def set_brush_type(self, brush_type: str):
        """Set current brush type"""
        if brush_type in BRUSH_TYPES:
            self.current_brush_type = brush_type

    def get_current_brush_color(self):
        """Get current brush color"""
        return BRUSH_TYPES[self.current_brush_type]["color"]

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.paint_pixmap:
            self.is_painting = True
            self.last_pt = self.mapToScene(ev.position().toPoint())
            self._draw_circle(self.last_pt)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pt = self.mapToScene(ev.position().toPoint())
        if self.is_painting:
            self._draw_line(self.last_pt, pt)
            self.last_pt = pt
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.is_painting = False
        super().mouseReleaseEvent(ev)

    def _draw_circle(self, center: QPointF):
        """Draw a circle directly on the main layer without any overlay."""
        painter = QPainter(self.paint_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setBrush(QBrush(self.get_current_brush_color()))
        painter.setPen(Qt.PenStyle.NoPen)
        r = self.brush_size / 2
        painter.drawEllipse(center, r, r)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)

    def _draw_line(self, p0: QPointF, p1: QPointF):
        """Draw continuous points between two points with circular 'brush' to ensure line continuity and consistent color."""
        dist = (p1 - p0).manhattanLength()
        step = max(1, self.brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps+1):
            t = i/steps
            pt = QPointF(p0.x() + (p1.x()-p0.x())*t,
                         p0.y() + (p1.y()-p0.y())*t)
            self._draw_circle(pt)

    def clear_painting(self):
        """Clear all strokes with one click."""
        if self.paint_pixmap:
            self.paint_pixmap.fill(Qt.GlobalColor.transparent)
            self.paint_item.setPixmap(self.paint_pixmap)

    def get_mask_pixmap(self):
        """Return grayscale mask with different pixel values for different brush types."""
        if not self.paint_pixmap:
            return None
        
        img = self.paint_pixmap.toImage()
        h, w = img.height(), img.width()
        
        # Create grayscale mask
        mask = QImage(w, h, QImage.Format.Format_Grayscale8)
        mask.fill(0)  # Fill with black (background)
        
        for y in range(h):
            for x in range(w):
                color = img.pixelColor(x, y)
                if color.alpha() > 0:  # Areas with drawing content
                    # Determine brush type based on color
                    brush_type = self._identify_brush_type(color)
                    if brush_type:
                        pixel_value = BRUSH_TYPES[brush_type]["pixel_value"]
                        mask.setPixel(x, y, pixel_value)
        
        return QPixmap.fromImage(mask)
    
    def _identify_brush_type(self, color):
        """Identify brush type based on color"""
        # Compare with known brush colors (ignoring alpha)
        for brush_type, brush_info in BRUSH_TYPES.items():
            brush_color = brush_info["color"]
            if (abs(color.red() - brush_color.red()) < 10 and
                abs(color.green() - brush_color.green()) < 10 and
                abs(color.blue() - brush_color.blue()) < 10):
                return brush_type
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataInfo = DataInfo()
        self.setWindowTitle(f"Circular Brush Annotation Tool: {self.dataInfo.test_name} (Test {self.dataInfo.test_id})")

        # Set unified font
        self.setup_fonts()
        
        # 1) Load CSV
        with open(self.dataInfo.meta_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            self.rows = list(reader)
            
        # 2) Find status and path columns
        candidates = [c for c in self.fieldnames
                      if c.lower() in
                         ('status','mask_status','label_status','annotation_status','annot_status')]
        if not candidates:
            QMessageBox.critical(self, "Error",
                "Cannot find a status column in CSV. "
                "Add a column named mask_status.")
            sys.exit(1)
        self.status_col = candidates[0]
        
        if 'relative_path' in self.fieldnames:
            self.path_col = 'relative_path'
        elif 'filename' in self.fieldnames:
            self.path_col = 'filename'
        else:
            QMessageBox.critical(self, "Error",
                "Cannot find image path column in CSV. "
                "Add 'relative_path' or 'filename'.")
            sys.exit(1)
            
        # 3) Build pending list - only process images with ND status
        self.pending = [i for i,r in enumerate(self.rows) if r[self.status_col] == STATUS_ND]
        if not self.pending:
            QMessageBox.information(self, "Completed", "No images to process (status is ND).")
            sys.exit(0)
        self.cur = 0

        # ─── UI Construction ─────────────────────────────────────────────────────────────
        w = QWidget()
        self.setCentralWidget(w)
        vlay = QVBoxLayout(w)

        # Drawing canvas
        self.view = PaintView()
        vlay.addWidget(self.view, stretch=1)

        # Brush type selection
        brush_type_layout = QHBoxLayout()
        brush_type_label = QLabel("Brush Type:")
        brush_type_label.setFont(self.chinese_font)
        brush_type_layout.addWidget(brush_type_label)
        
        self.brush_type_combo = QComboBox()
        for brush_type, brush_info in BRUSH_TYPES.items():
            self.brush_type_combo.addItem(brush_info["display_name"], brush_type)
        self.brush_type_combo.currentIndexChanged.connect(self.on_brush_type_changed)
        brush_type_layout.addWidget(self.brush_type_combo)
        brush_type_layout.addStretch()  # Add stretch to push items to left
        
        vlay.addLayout(brush_type_layout)

        # Brush size control
        brush_layout = QHBoxLayout()
        brush_label = QLabel("Brush Size:")
        brush_label.setFont(self.chinese_font)
        brush_layout.addWidget(brush_label)
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setMinimum(MIN_BRUSH_SIZE)
        self.brush_slider.setMaximum(MAX_BRUSH_SIZE)
        self.brush_slider.setValue(DEFAULT_BRUSH_SIZE)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(self.brush_slider)
        
        self.brush_spinbox = QSpinBox()
        self.brush_spinbox.setMinimum(MIN_BRUSH_SIZE)
        self.brush_spinbox.setMaximum(MAX_BRUSH_SIZE)
        self.brush_spinbox.setValue(DEFAULT_BRUSH_SIZE)
        self.brush_spinbox.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(self.brush_spinbox)
        
        vlay.addLayout(brush_layout)

        # Control buttons
        hlay = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_clear = QPushButton("Clear Drawing")
        self.btn_pass = QPushButton("Skip")
        self.btn_test = QPushButton("Save as Test")
        self.btn_true = QPushButton("Save as True")
        
        # Set button styles and fonts
        buttons = [self.btn_prev, self.btn_next, self.btn_clear, 
                  self.btn_test, self.btn_pass, self.btn_true]
        for btn in buttons:
            btn.setFont(self.chinese_font)
        
        hlay.addWidget(self.btn_prev)
        hlay.addWidget(self.btn_next)
        hlay.addWidget(self.btn_clear)
        hlay.addWidget(self.btn_pass)
        hlay.addWidget(self.btn_test)
        hlay.addWidget(self.btn_true)
        vlay.addLayout(hlay)

        # Status display and log area
        self.status_label = QLabel()
        self.status_label.setFont(self.chinese_font)
        vlay.addWidget(self.status_label)
        
        # Add log display area
        self.log_area = QTextEdit()
        self.log_area.setFont(self.chinese_font)
        self.log_area.setMaximumHeight(120)  # Limit height
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        vlay.addWidget(self.log_area)
        
        help_text = QLabel(
            "Operation Tips:\n"
            "• Left-click and drag to draw area with selected brush type\n"
            "• Use slider or spin box to adjust brush size\n"
            "• Clear Drawing: Clear current drawing content\n"
            "• Save as Test: Save current drawing state with 'test' status\n"
            "• Save as True: Save current drawing state with 'true' status\n"
            "• Skip: Don't save anything, mark as 'pass'"
        )
        help_text.setFont(self.chinese_font)
        help_text.setStyleSheet("""
            QLabel {
                color: black; 
                font-size: 13px; 
                padding: 10px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        vlay.addWidget(help_text)

        # Connect signals
        self.btn_prev.clicked.connect(self.goto_prev)
        self.btn_next.clicked.connect(self.goto_next)
        self.btn_clear.clicked.connect(self.clear_current)
        self.btn_test.clicked.connect(self.save_test_sample)
        self.btn_pass.clicked.connect(self.mark_pass)
        self.btn_true.clicked.connect(self.save_true_sample)

        # Synchronize slider and spin box
        self.brush_slider.valueChanged.connect(self.brush_spinbox.setValue)
        self.brush_spinbox.valueChanged.connect(self.brush_slider.setValue)

        # Initialize log
        self.log("Annotation tool started")
        
        # Load first image
        self.load_current()

    def on_brush_type_changed(self, index):
        """Handle brush type change"""
        brush_type = self.brush_type_combo.itemData(index)
        if brush_type and brush_type in BRUSH_TYPES:
            self.view.set_brush_type(brush_type)
            brush_info = BRUSH_TYPES[brush_type]
            self.log(f"Switched to {brush_info['display_name']} brush (pixel value: {brush_info['pixel_value']})")

    def log(self, message):
        """Add message to log area"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_area.append(formatted_msg)
        # Auto scroll to bottom
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

    def setup_fonts(self):
        """Set unified font"""
        self.chinese_font = QFont()
        chinese_fonts = "SimSun"
        self.chinese_font.setFamily(chinese_fonts)
        self.chinese_font.setPointSize(10)
        self.chinese_font.setWeight(QFont.Weight.Normal)
        
        # Set application default font
        QApplication.instance().setFont(self.chinese_font)

    def on_brush_size_changed(self, value):
        """Handle brush size change"""
        self.view.set_brush_size(value)

    def load_current(self):
        """Load current image"""
        idx = self.pending[self.cur]
        row = self.rows[idx]
        # Parse path
        p = Path(row[self.path_col])
        if not p.exists():
            p = self.dataInfo.images_dir / Path(row[self.path_col]).name
        if not p.exists():
            QMessageBox.critical(self, "Error", f"Image not found:\n{row[self.path_col]}")
            sys.exit(1)
        self.current_path = p
        self.view.load_image(p)
        
        # Display detailed status information
        status_text = f"[{self.cur+1}/{len(self.pending)}] {p.name} — Status: {row[self.status_col]}"
        self.status_label.setText(status_text)
        
        # Log record
        self.log(f"Load image: {p.name}")
        
        # Update button states based on current status
        self.update_button_states()

    def update_button_states(self):
        """Update button availability based on current status"""
        # Navigation buttons
        self.btn_prev.setEnabled(self.cur > 0)
        self.btn_next.setEnabled(self.cur < len(self.pending) - 1)

    def clear_current(self):
        """Clear current drawing content"""
        self.view.clear_painting()
        self.log("Drawing content cleared")

    def goto_prev(self):
        if self.cur > 0:
            self.cur -= 1
            self.load_current()

    def goto_next(self):
        if self.cur < len(self.pending) - 1:
            self.cur += 1
            self.load_current()

    def mark_pass(self):
        """Skip this image"""
        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = STATUS_PASS
        self._write_csv()
        
        file_name = self.current_path.name
        self.log(f"Skip image: {file_name} (status: pass)")
        
        self._remove_current()

    def save_test_sample(self):
        """Save test sample (current drawing state as mask)"""
        # Get image size
        if not self.view.pixmap_item.pixmap():
            self.log("Error: No image loaded")
            return
            
        img_size = self.view.pixmap_item.pixmap().size()
        
        # Get current mask (could be all-black if nothing drawn, or contain drawings)
        mask_pixmap = self.view.get_mask_pixmap()
        if mask_pixmap:
            # Has drawing content
            mask_image = mask_pixmap.toImage()
        else:
            # No drawing content, create all-black mask
            mask_image = QImage(img_size, QImage.Format.Format_Grayscale8)
            mask_image.fill(0)  # All black (background)
        
        # Save and update status
        if self._save_mask_image(mask_image, STATUS_TEST):
            file_name = self.current_path.name
            has_content = mask_pixmap is not None
            content_desc = "with drawing content" if has_content else "all-black mask"
            self.log(f"Save test sample: {file_name} ({content_desc})")
            self._remove_current()

    def save_true_sample(self):
        """Save true sample (current drawing state as mask)"""
        # Get image size
        if not self.view.pixmap_item.pixmap():
            self.log("Error: No image loaded")
            return
            
        img_size = self.view.pixmap_item.pixmap().size()
        
        # Get current mask (could be all-black if nothing drawn, or contain drawings)
        mask_pixmap = self.view.get_mask_pixmap()
        if mask_pixmap:
            # Has drawing content
            mask_image = mask_pixmap.toImage()
        else:
            # No drawing content, create all-black mask
            mask_image = QImage(img_size, QImage.Format.Format_Grayscale8)
            mask_image.fill(0)  # All black (background)

        # Save and update status
        if self._save_mask_image(mask_image, STATUS_TRUE):
            file_name = self.current_path.name
            has_content = mask_pixmap is not None
            content_desc = "with drawing content" if has_content else "all-black mask"
            self.log(f"Save true sample: {file_name} ({content_desc})")
            self._remove_current()

    def _save_mask_image(self, mask_image, status):
        """Save mask image and update CSV status"""
        # Generate save filename
        base_name = self.current_path.stem
        out_name = f"{base_name}_mask.png"
        out_path = self.dataInfo.masks_dir / out_name
        
        # Save mask
        success = mask_image.save(str(out_path))
        if not success:
            self.log(f"Save failed: Unable to save mask to {out_path}")
            return False

        # Update CSV
        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = status
        
        # Add mask path column (if not exists)
        if 'mask_path' not in self.fieldnames:
            self.fieldnames.append('mask_path')
        
        # Use relative path
        try:
            self.rows[idx]['mask_path'] = str(out_path)
        except ValueError:
            # print error in log box
            self.log(f"Error: Unable to set mask path {out_path} (row {idx})")
            
        self._write_csv()
        return True

    def _remove_current(self):
        """Remove current item after processing and continue"""
        self.pending.pop(self.cur)
        remaining = len(self.pending)
        
        if self.cur >= remaining:
            self.cur = remaining - 1
            
        if not self.pending:
            self.log("All completed: All image annotation completed!")
            QMessageBox.information(self, "All Completed", "All image annotation completed!")
            QApplication.quit()
            return
            
        self.log(f"Remaining images to process: {remaining}")
        self.load_current()

    def _write_csv(self):
        """Write CSV file"""
        with open(self.dataInfo.meta_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())