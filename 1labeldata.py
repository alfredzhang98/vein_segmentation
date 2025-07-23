#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6-based 圆形刷子标注工具 for annotating phantom images.
功能特点：
- 圆形刷子自由绘制
- 可调节刷子大小
- 实时红色显示绘制区域
- 保存二值化mask图像
- 支持positive/negative/pass/ND四种状态标记
- 修复颜色叠加问题
- 修复mask颜色问题和简化操作流程
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
    QGraphicsScene, QGraphicsPixmapItem, QSlider, QSpinBox, QTextEdit
)

# ─── Configuration ────────────────────────────────────────────────────────────
test_id    = "1"
test_name  = "phantom_taobao"
base_dir   = Path("data") / test_name
csv_path   = base_dir / f"meta_{test_name}_{test_id}.csv"
img_dir    = base_dir / "images"
mask_dir   = base_dir / "masks"
mask_dir.mkdir(parents=True, exist_ok=True)

# 正在加载测试数据: phantom_taobao (Test 1)
# CSV路径: data\phantom_taobao\meta_phantom_taobao_1.csv
# 图像目录: data\phantom_taobao\images
# Mask目录: data\phantom_taobao\masks

print(f"正在加载测试数据: {test_name} (Test {test_id})")
print(f"CSV路径: {csv_path}")
print(f"图像目录: {img_dir}")
print(f"Mask目录: {mask_dir}")


# 配置参数
MIN_BRUSH_SIZE = 5   # 最小刷子大小
MAX_BRUSH_SIZE = 100 # 最大刷子大小
DEFAULT_BRUSH_SIZE = 20  # 默认刷子大小
BRUSH_COLOR = QColor(0, 255, 0, 50)  # 绿色半透明

# 状态常量
STATUS_ND = "ND"           # 待处理
STATUS_POSITIVE = "positive"  # 正样本（有标注内容）
STATUS_NEGATIVE = "negative"  # 负样本（空白标注）
STATUS_PASS = "pass"         # 跳过
# ────────────────────────────────────────────────────────────────────────────────

class PaintView(QGraphicsView):
    """用Source直接绘制的圆形刷子，抗锯齿+不叠加颜色。"""
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))

        # 底图
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        # 绘制层
        self.paint_pixmap = None
        self.paint_item = QGraphicsPixmapItem()
        self.scene().addItem(self.paint_item)

        self.brush_size = DEFAULT_BRUSH_SIZE
        self.is_painting = False
        self.last_pt = QPointF()

        self.setMouseTracking(True)

    def load_image(self, path: Path):
        pix = QPixmap(str(path))
        self.pixmap_item.setPixmap(pix)
        self.setSceneRect(pix.rect())

        # 初始化一个透明的画布
        self.paint_pixmap = QPixmap(pix.size())
        self.paint_pixmap.fill(Qt.GlobalColor.transparent)
        self.paint_item.setPixmap(self.paint_pixmap)

    def set_brush_size(self, s: int):
        self.brush_size = s

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
        """在主层直接画一个圆，不做任何叠加。"""
        painter = QPainter(self.paint_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setBrush(QBrush(BRUSH_COLOR))
        painter.setPen(Qt.PenStyle.NoPen)
        r = self.brush_size / 2
        painter.drawEllipse(center, r, r)
        painter.end()
        self.paint_item.setPixmap(self.paint_pixmap)

    def _draw_line(self, p0: QPointF, p1: QPointF):
        """在两点间用圆形"刷子"画连续点，保证线连续且颜色一致。"""
        dist = (p1 - p0).manhattanLength()
        step = max(1, self.brush_size / 2)
        steps = int(dist / step) + 1
        for i in range(steps+1):
            t = i/steps
            pt = QPointF(p0.x() + (p1.x()-p0.x())*t,
                         p0.y() + (p1.y()-p0.y())*t)
            self._draw_circle(pt)

    def clear_painting(self):
        """一键清除所有笔划。"""
        if self.paint_pixmap:
            self.paint_pixmap.fill(Qt.GlobalColor.transparent)
            self.paint_item.setPixmap(self.paint_pixmap)

    def get_mask_pixmap(self):
        """直接返回二值Mask：将半透明区变白，其它黑。"""
        if not self.paint_pixmap:
            return None
        
        img = self.paint_pixmap.toImage()
        h, w = img.height(), img.width()
        
        # 修复：使用 RGB32 格式创建mask，然后转换为灰度
        mask = QImage(w, h, QImage.Format.Format_RGB32)
        mask.fill(Qt.GlobalColor.black)  # 默认填充黑色
        
        for y in range(h):
            for x in range(w):
                color = img.pixelColor(x, y)
                if color.alpha() > 0:  # 有绘制内容的地方
                    mask.setPixel(x, y, QColor(255, 255, 255).rgb())  # 设置为纯白色
        
        return QPixmap.fromImage(mask)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"圆形刷子标注工具: {test_name} (Test {test_id})")
        
        # 设置统一的中文字体
        self.setup_fonts()
        
        # 1) 加载CSV
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            self.rows = list(reader)
            
        # 2) 查找状态和路径列
        candidates = [c for c in self.fieldnames
                      if c.lower() in
                         ('status','mask_status','label_status','annotation_status','annot_status')]
        if not candidates:
            QMessageBox.critical(self, "Error",
                "Cannot find a status column in CSV. "
                "Add a column named mask_status (values ND/positive/negative/pass).")
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
            
        # 3) 构建待处理列表 - 只处理状态为ND的图像
        self.pending = [i for i,r in enumerate(self.rows) if r[self.status_col] == STATUS_ND]
        if not self.pending:
            QMessageBox.information(self, "完成", "没有待处理的图像（状态为ND）。")
            sys.exit(0)
        self.cur = 0

        # ─── UI构建 ─────────────────────────────────────────────────────────────
        w = QWidget()
        self.setCentralWidget(w)
        vlay = QVBoxLayout(w)

        # 绘制画布
        self.view = PaintView()
        vlay.addWidget(self.view, stretch=1)

        # 刷子大小控制
        brush_layout = QHBoxLayout()
        brush_label = QLabel("刷子大小:")
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

        # 控制按钮
        hlay = QHBoxLayout()
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.btn_clear = QPushButton("清除绘制")
        self.btn_pass = QPushButton("跳过")
        self.btn_negative = QPushButton("负样本")
        self.btn_positive = QPushButton("正样本")
        
        # 设置按钮样式和字体
        buttons = [self.btn_prev, self.btn_next, self.btn_clear, 
                  self.btn_negative, self.btn_pass, self.btn_positive]
        for btn in buttons:
            btn.setFont(self.chinese_font)
        
        hlay.addWidget(self.btn_prev)
        hlay.addWidget(self.btn_next)
        hlay.addWidget(self.btn_clear)
        hlay.addWidget(self.btn_pass)
        hlay.addWidget(self.btn_negative)
        hlay.addWidget(self.btn_positive)
        vlay.addLayout(hlay)

        # 状态显示和日志区域
        self.status_label = QLabel()
        self.status_label.setFont(self.chinese_font)
        vlay.addWidget(self.status_label)
        
        # 添加日志显示区域
        self.log_area = QTextEdit()
        self.log_area.setFont(self.chinese_font)
        self.log_area.setMaximumHeight(120)  # 限制高度
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
            "操作提示：\n"
            "• 左键按住拖拽绘制区域（已修复颜色叠加问题）\n"
            "• 滑动条或数字框调节刷子大小\n"
            "• 清除绘制：清空当前绘制内容\n"
            "• 负样本：保存全黑mask（无标注区域）\n"
            "• 正样本：保存绘制区域的二值化mask（有标注区域）\n"
            "• 跳过：不保存任何内容，标记为pass"
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

        # 连接信号
        self.btn_prev.clicked.connect(self.goto_prev)
        self.btn_next.clicked.connect(self.goto_next)
        self.btn_clear.clicked.connect(self.clear_current)
        self.btn_negative.clicked.connect(self.save_negative_sample)
        self.btn_pass.clicked.connect(self.mark_pass)
        self.btn_positive.clicked.connect(self.save_positive_sample)

        # 同步滑动条和数字框
        self.brush_slider.valueChanged.connect(self.brush_spinbox.setValue)
        self.brush_spinbox.valueChanged.connect(self.brush_slider.setValue)

        # 初始化日志
        self.log("标注工具已启动")
        
        # 加载第一张图像
        self.load_current()

    def log(self, message):
        """在日志区域添加消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_area.append(formatted_msg)
        # 自动滚动到底部
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

    def setup_fonts(self):
        """设置统一的中文字体"""
        self.chinese_font = QFont()
        chinese_fonts = "SimSun"
        self.chinese_font.setFamily(chinese_fonts)
        self.chinese_font.setPointSize(10)
        self.chinese_font.setWeight(QFont.Weight.Normal)
        
        # 设置应用程序默认字体
        QApplication.instance().setFont(self.chinese_font)

    def on_brush_size_changed(self, value):
        """刷子大小改变时的处理"""
        self.view.set_brush_size(value)

    def load_current(self):
        """加载当前图像"""
        idx = self.pending[self.cur]
        row = self.rows[idx]
        # 解析路径
        p = Path(row[self.path_col])
        if not p.exists():
            p = img_dir / Path(row[self.path_col]).name
        if not p.exists():
            QMessageBox.critical(self, "Error", f"Image not found:\n{row[self.path_col]}")
            sys.exit(1)
        self.current_path = p
        self.view.load_image(p)
        
        # 显示更详细的状态信息
        status_text = f"[{self.cur+1}/{len(self.pending)}] {p.name} — 状态: {row[self.status_col]}"
        self.status_label.setText(status_text)
        
        # 记录日志
        self.log(f"加载图像: {p.name}")
        
        # 根据当前状态更新按钮可用性
        self.update_button_states()

    def update_button_states(self):
        """根据当前状态更新按钮可用性"""
        # 导航按钮
        self.btn_prev.setEnabled(self.cur > 0)
        self.btn_next.setEnabled(self.cur < len(self.pending) - 1)

    def clear_current(self):
        """清除当前绘制内容"""
        self.view.clear_painting()
        self.log("已清除绘制内容")

    def goto_prev(self):
        if self.cur > 0:
            self.cur -= 1
            self.load_current()

    def goto_next(self):
        if self.cur < len(self.pending) - 1:
            self.cur += 1
            self.load_current()

    def mark_pass(self):
        """跳过这张图像"""
        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = STATUS_PASS
        self._write_csv()
        
        file_name = self.current_path.name
        self.log(f"跳过图像: {file_name} (状态: pass)")
        
        self._remove_current()

    def save_negative_sample(self):
        """保存负样本（全黑mask）"""
        # 获取图像尺寸
        if not self.view.pixmap_item.pixmap():
            self.log("错误: 没有加载图像")
            return
            
        img_size = self.view.pixmap_item.pixmap().size()
        
        # 创建全黑mask
        mask_image = QImage(img_size, QImage.Format.Format_RGB32)
        mask_image.fill(Qt.GlobalColor.black)  # 全黑
        
        # 保存并更新状态
        if self._save_mask_image(mask_image, STATUS_NEGATIVE):
            file_name = self.current_path.name
            self.log(f"保存负样本: {file_name} (全黑mask)")
            self._remove_current()

    def save_positive_sample(self):
        """保存正样本（有绘制内容的mask）"""
        mask_pixmap = self.view.get_mask_pixmap()
        if not mask_pixmap:
            # 没有绘制内容，询问是否保存为负样本
            file_name = self.current_path.name
            self.log(f"无绘制内容: {file_name} - 自动保存为负样本")
            self.save_negative_sample()
            return

        # 保存并更新状态
        if self._save_mask_image(mask_pixmap.toImage(), STATUS_POSITIVE):
            file_name = self.current_path.name
            self.log(f"保存正样本: {file_name} (包含绘制区域)")
            self._remove_current()

    def _save_mask_image(self, mask_image, status):
        """保存mask图像并更新CSV状态"""
        # 生成保存文件名
        base_name = self.current_path.stem
        out_name = f"{base_name}_mask.png"
        out_path = mask_dir / out_name
        
        # 保存mask
        success = mask_image.save(str(out_path))
        if not success:
            self.log(f"保存失败: 无法保存mask到 {out_path}")
            return False

        # 更新CSV
        idx = self.pending[self.cur]
        self.rows[idx][self.status_col] = status
        
        # 添加mask路径列（如果不存在）
        if 'mask_path' not in self.fieldnames:
            self.fieldnames.append('mask_path')
        
        # 使用相对路径
        try:
            self.rows[idx]['mask_path'] = str(out_path)
        except ValueError:
            # print error in log box
            self.log(f"错误: 无法设置mask路径 {out_path} (行 {idx})")
            
        self._write_csv()
        return True

    def _remove_current(self):
        """处理完当前图像后移除并继续"""
        self.pending.pop(self.cur)
        remaining = len(self.pending)
        
        if self.cur >= remaining:
            self.cur = remaining - 1
            
        if not self.pending:
            self.log("全部完成: 所有图像标注完成！")
            QMessageBox.information(self, "全部完成", "所有图像标注完成！")
            QApplication.quit()
            return
            
        self.log(f"剩余待处理图像: {remaining} 张")
        self.load_current()

    def _write_csv(self):
        """写入CSV文件"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())