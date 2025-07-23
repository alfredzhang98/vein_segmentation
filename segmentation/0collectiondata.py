#!/usr/bin/env python3

import sys
import csv
import ctypes
import time
import os.path
from pathlib import Path
from datetime import datetime
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot
from PIL import Image

# Load Clarius libraries
if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")
else:
    import pyclariuscast

# Configuration
TEST_ID = "2"
TEST_NAME = "phantom_taobao"
OUTPUT_DIR = Path(f"data")
IP = "192.168.1.1"
PORT = 5828

# Clarius userFunction commands
CMD_FREEZE = 1
CMD_CAPTURE_IMAGE = 2
CMD_DEPTH_DEC = 4
CMD_DEPTH_INC = 5
CMD_GAIN_DEC = 6
CMD_GAIN_INC = 7

# Custom events
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen

class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))

# Signaller
class Signaller(QtCore.QObject):
    freeze = QtCore.Signal(bool)
    image = QtCore.Signal(QtGui.QImage)
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.usimage = QtGui.QImage()
        self.original_image = QtGui.QImage()
        self.micropixel = 0.0
        self.timestamp = 0.0
        self.qw = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.bpp = 0.0

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True

signaller = Signaller()

class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, cast):
        QtWidgets.QGraphicsView.__init__(self)
        self.cast = cast
        self.setScene(QtWidgets.QGraphicsScene())
        # 初始化一个空图像
        self.image = QtGui.QImage(640, 480, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.black)

    def updateImage(self, img):
        self.image = img
        self.scene().invalidate()

    def resizeEvent(self, evt):
        w = evt.size().width()
        h = evt.size().height()
        try:
            self.cast.setOutputSize(w, h)
            self.image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
            self.image.fill(QtCore.Qt.black)
            self.setSceneRect(0, 0, w, h)
        except Exception as e:
            print(f"Resize error: {e}")

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QtCore.Qt.black)

    def drawForeground(self, painter, rect):
        if hasattr(self, 'image') and not self.image.isNull():
            painter.drawImage(rect, self.image)

# 自定义保存预览对话框
class SavePreviewDialog(QtWidgets.QDialog):
    def __init__(self, image, filename, image_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Image Preview")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # 文件名标签
        filename_label = QtWidgets.QLabel(f"Save as: {filename}")
        filename_label.setAlignment(Qt.AlignCenter)
        filename_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 10px;")
        layout.addWidget(filename_label)
        
        # 图片预览
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        
        # 缩放图片以适应预览窗口
        max_size = QtCore.QSize(500, 350)
        scaled_pixmap = QtGui.QPixmap.fromImage(image).scaled(
            max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setMinimumSize(500, 350)
        
        layout.addWidget(self.image_label)
        
        # 图片信息
        info_layout = QtWidgets.QVBoxLayout()
        info_text1 = f"Original Size: {image.width()}×{image.height()} | Format: {image.format().name}"
        info_label1 = QtWidgets.QLabel(info_text1)
        info_label1.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_label1)
        layout.addLayout(info_layout)
        
        # 按钮
        button_layout = QtWidgets.QHBoxLayout()
        self.yes_button = QtWidgets.QPushButton("Yes, Save")
        self.no_button = QtWidgets.QPushButton("No, Cancel")
        
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)
        layout.addLayout(button_layout)
        
        # 连接信号
        self.yes_button.clicked.connect(self.accept)
        self.no_button.clicked.connect(self.reject)

# Main GUI
class MainWidget(QtWidgets.QMainWindow):
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.test_id = TEST_ID
        self.output_dir = OUTPUT_DIR
        self.img_dir = self.output_dir / f"{TEST_NAME}" / "images"
        self.meta_file = self.output_dir / f"{TEST_NAME}/meta_{TEST_NAME}_{self.test_id}.csv"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_file.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.meta_file, 'w', newline='') as f:
                csv.writer(f).writerow(["id","filename","relative_path",
                                        "raw_timestamp","timestamp",
                                        "depth","gain","frequency",
                                        "bpp", "image_width","image_height", "micropixel",
                                        # "qw", "qx", "qy", "qz", 
                                        "test_name", "test_id", "mask_status"])

        with open(self.meta_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.saved_count = max(0, len(rows) - 1)

        # 初始化参数
        self.depth_counter = 0     # 本地计数器
        self.gain_counter = 0      # 本地计数器
        self.depth_step = 1        # 深度步长
        self.gain_step = 10        # 增益步长
        self.actual_depth = 4      # 实际设备值
        self.actual_gain = 50      # 实际设备值
        self.frequency = "8/10.0"  # 频率
        self.frozen = True

        # key 防止重复按键
        self.key_last_pressed = {
            Qt.Key_1: 0,  # Depth increase
            Qt.Key_2: 0,  # Depth decrease  
            Qt.Key_3: 0,  # Gain increase
            Qt.Key_4: 0,  # Gain decrease
            Qt.Key_S: 0   # Save
        }
        self.key_interval = 2.0

        self.setup_ui()
        self.connect_signals()
        self.initialize_cast()

    def setup_ui(self):
        # UI
        self.setWindowTitle(f"Clarius Capture - Test {self.test_id}")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 连接控制
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.ip_edit = QtWidgets.QLineEdit(IP)
        self.port_edit = QtWidgets.QLineEdit(str(PORT))
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_freeze = QtWidgets.QPushButton("Run")
        self.btn_freeze.setEnabled(False)
        self.btn_quit = QtWidgets.QPushButton("Quit")
        
        ctrl_layout.addWidget(QtWidgets.QLabel("IP:"))
        ctrl_layout.addWidget(self.ip_edit)
        ctrl_layout.addWidget(QtWidgets.QLabel("Port:"))
        ctrl_layout.addWidget(self.port_edit)
        ctrl_layout.addWidget(self.btn_connect)
        ctrl_layout.addWidget(self.btn_freeze)
        ctrl_layout.addWidget(self.btn_quit)
        layout.addLayout(ctrl_layout)

        # 参数显示
        param_layout = QtWidgets.QHBoxLayout()
        self.depth_label = QtWidgets.QLabel(f"Depth: ({self.actual_depth} cm)")
        self.gain_label = QtWidgets.QLabel(f"Gain: ({self.actual_gain}%)")
        self.freq_label = QtWidgets.QLabel(f"Frequency: {self.frequency} MHz")
        self.depth_label.setStyleSheet("padding: 2px; border: 1px solid gray;")
        self.gain_label.setStyleSheet("padding: 2px; border: 1px solid gray;")
        self.freq_label.setStyleSheet("padding: 2px; border: 1px solid gray;")

        param_layout.addWidget(self.depth_label)
        param_layout.addWidget(self.gain_label)
        param_layout.addWidget(self.freq_label)
        layout.addLayout(param_layout)

        # Image view
        self.img = ImageView(self.cast)
        layout.addWidget(self.img)

        # 说明
        help_layout = QtWidgets.QVBoxLayout()

        help_label1 = QtWidgets.QLabel("Controls: 1 increase Depth | 2 decrease Depth | 3 increase Gain | 4 decrease Gain | S Save | Q Quit")
        help_label2 = QtWidgets.QLabel(f"Please Setting: Depth default is {self.actual_depth} cm, step is {self.depth_step} cm; Gain default is {self.actual_gain}%, step is {self.gain_step}%.")
        help_label1.setAlignment(Qt.AlignCenter)
        help_label2.setAlignment(Qt.AlignCenter)
        help_label1.setStyleSheet("font-weight: bold;")
        help_label2.setStyleSheet("font-style: italic; color: red; font-weight: bold;")
        help_layout.addWidget(help_label1)
        help_layout.addWidget(help_label2)
        layout.addLayout(help_layout)

        # Status bar
        self.statusBar().showMessage("Disconnected")

    def connect_signals(self):
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_freeze.clicked.connect(self.toggle_freeze)
        self.btn_quit.clicked.connect(self.shutdown)
        signaller.freeze.connect(self.on_freeze)
        signaller.image.connect(self.on_image)

    def initialize_cast(self):
        path = os.path.expanduser("~/")
        try:
            if self.cast.init(path, 640, 480):
                self.statusBar().showMessage("Initialized - Ready to connect")
            else:
                self.statusBar().showMessage("Failed to initialize")
        except Exception as e:
            self.statusBar().showMessage(f"Init error: {e}")

    def update_parameter_display(self):
        """更新参数显示"""
        depth = self.actual_depth + self.depth_counter * self.depth_step
        gain = self.actual_gain + self.gain_counter * self.gain_step

        depth_text = f"Depth: ({depth} cm)"
        gain_text = f"Gain: ({gain}%)"
        freq_text = f"Frequency: {self.frequency} MHz"

        self.depth_label.setText(depth_text)
        self.gain_label.setText(gain_text)
        self.freq_label.setText(freq_text)

    def toggle_connection(self):
        if not self.cast.isConnected():
            try:
                if self.cast.connect(self.ip_edit.text(), int(self.port_edit.text()), "research"):
                    self.statusBar().showMessage("Connected")
                    self.btn_connect.setText("Disconnect")
                    self.btn_freeze.setEnabled(True)
                else:
                    self.statusBar().showMessage("Connect Failed")
            except Exception as e:
                self.statusBar().showMessage(f"Connect error: {e}")
        else:
            try:
                self.cast.disconnect()
                self.statusBar().showMessage("Disconnected")
                self.btn_connect.setText("Connect")
                self.btn_freeze.setEnabled(False)
                self.img.scene().clear()
                self.btn_freeze.setText("Run")
                self.frozen = True
                self.update_parameter_display()
            except Exception as e:
                self.statusBar().showMessage(f"Disconnect error: {e}")

    def toggle_freeze(self):
        if not self.cast.isConnected(): 
            return
        try:
            self.cast.userFunction(CMD_FREEZE, 0)
        except Exception as e:
            self.statusBar().showMessage(f"Freeze error: {e}")

    @Slot(bool)
    def on_freeze(self, frozen):
        self.frozen = frozen
        if frozen:
            self.btn_freeze.setText("Run")
            self.statusBar().showMessage("Paused")
        else:
            self.btn_freeze.setText("Freeze")
            self.statusBar().showMessage("Streaming")

    @Slot(QtGui.QImage)
    def on_image(self, img):
        self.img.updateImage(img)

    def keyPressEvent(self, event):
        if not self.cast.isConnected(): 
            return
        
        key = event.key()
        current_time = time.time()

        # 检查是否是需要防抖的按键
        if key in self.key_last_pressed:
            # 检查时间间隔
            if current_time - self.key_last_pressed[key] < self.key_interval:
                remaining_time = self.key_interval - (current_time - self.key_last_pressed[key])
                key_name = {
                    Qt.Key_1: "Depth+",
                    Qt.Key_2: "Depth-", 
                    Qt.Key_3: "Gain+",
                    Qt.Key_4: "Gain-",
                    Qt.Key_S: "Save"
                }[key]
                self.statusBar().showMessage(f"{key_name}: Wait {remaining_time:.1f}s")
                return
            
            # 更新该按键的最后按下时间
            self.key_last_pressed[key] = current_time

        try:
            if key == Qt.Key_2:
                self.cast.userFunction(CMD_DEPTH_DEC, 0)
                self.depth_counter -= 1
                self.update_parameter_display()
                self.statusBar().showMessage("Depth decreased")
            elif key == Qt.Key_1:
                self.cast.userFunction(CMD_DEPTH_INC, 0)
                self.depth_counter += 1
                self.update_parameter_display()
                self.statusBar().showMessage("Depth increased")
            elif key == Qt.Key_4:
                self.cast.userFunction(CMD_GAIN_DEC, 0)
                self.gain_counter -= 1
                self.update_parameter_display()
                self.statusBar().showMessage("Gain decreased")
            elif key == Qt.Key_3:
                self.cast.userFunction(CMD_GAIN_INC, 0)
                self.gain_counter += 1
                self.update_parameter_display()
                self.statusBar().showMessage("Gain increased")
            elif key == Qt.Key_S and not self.frozen:
                self.save_image_with_preview()
                self.statusBar().showMessage("Image saved")
            elif key == Qt.Key_Q:
                self.shutdown()
        except Exception as e:
            self.statusBar().showMessage(f"Key error: {e}")

    def save_image_with_preview(self):
        """带预览的保存图片 - 使用原始尺寸图像"""
        # 使用原始图像而不是界面显示的图像
        if signaller.original_image.isNull():
            QtWidgets.QMessageBox.warning(self, "Warning", "No original image to save!")
            return
        
        # 准备图像信息
        # 读取原始长度和高度
        im = Image.fromqimage(signaller.original_image)
        image_info = f"Original: {im.width}×{im.height}"

        micropixel = signaller.micropixel
        raw_timestamp = signaller.timestamp
        # w = signaller.qw
        # x = signaller.qx
        # y = signaller.qy    
        # z = signaller.qz
        bpp = signaller.bpp

        depth = self.actual_depth + self.depth_counter * self.depth_step
        gain = self.actual_gain + self.gain_counter * self.gain_step
            
        # 生成文件名
        id = f"{self.saved_count:04d}"
        self.saved_count += 1
        ts = str(int(time.time() * 1000))
        name = f"{id}_{ts}_T_{self.test_id}.png"

        # 显示预览对话框
        dialog = SavePreviewDialog(signaller.original_image, name, image_info, self)
        
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            try:
                path = self.img_dir / name

                # Make sure the Alpha channel is handled correctly
                if im.mode != 'RGBA':
                    im = im.convert('RGBA')
                alpha = im.getchannel("A")
                bbox = alpha.getbbox()
                if bbox:
                    cropped_im = im.crop(bbox)
                    cropped_im.save(path, format='PNG')
                    width, height = cropped_im.size
                else:
                    im.save(path, format='PNG')
                    width, height = im.size

                
                # 保存元数据，包含真实的图像尺寸
                with open(self.meta_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        id, 
                        name, 
                        str(self.img_dir / name),
                        raw_timestamp,
                        ts,
                        depth,
                        gain,
                        self.frequency,
                        bpp,
                        width,
                        height,
                        micropixel,
                        # w,
                        # x,
                        # y,
                        # z,
                        TEST_NAME,
                        self.test_id,
                        "ND"  # Mask status is not done (ND)
                    ])
            
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {e}")
        else:
            self.statusBar().showMessage("Save cancelled")

    @Slot()
    def shutdown(self):
        try:
            if self.cast.isConnected():
                self.cast.disconnect()
            if sys.platform.startswith("linux"):
                ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
            self.cast.destroy()
        except Exception as e:
            print(f"Shutdown error: {e}")
        finally:
            QtWidgets.QApplication.quit()

# 修改后的回调函数
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    bpp = sz / (width * height)
    # print(f"Image size: {width}x{height}, BPP: {bpp}, Microns per pixel: {micronsPerPixel}, Timestamp: {timestamp}, Angle: {angle}")
    # print(f"IMU data: {imu[0].qw} {imu[0].qx} {imu[0].qy} {imu[0].qz}" if imu else "No IMU data")
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    
    signaller.original_image = img.copy()
    signaller.micropixel = micronsPerPixel
    signaller.timestamp = timestamp
    # signaller.qw = imu[0].qw if imu else 0.0
    # signaller.qx = imu[0].qx if imu else 0.0
    # signaller.qy = imu[0].qy if imu else 0.0
    # signaller.qz = imu[0].qz if imu else 0.0
    signaller.bpp = bpp
    signaller.usimage = img.copy()
    
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return

def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return

def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return

def newImuData(imu):
    return

def freezeFn(frozen):
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return

def buttonsFn(button, clicks):
    return

# Main
if __name__ == '__main__':
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWidget(cast)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())