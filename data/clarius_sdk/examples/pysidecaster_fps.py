#!/usr/bin/env python

import ctypes
import os.path
import sys
import time
from pathlib import Path
from typing import Final
from threading import Lock

if sys.platform.startswith("linux"):
    libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle
    pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")

import pyclariuscast
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot, QThread, QTimer, QMutex, QMutexLocker
from PySide6.QtWidgets import QSplitter, QGroupBox, QGridLayout

# Constants
CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14


class ImageProcessor(QThread):
    """独立的图像处理线程"""
    imageReady = Signal(QtGui.QImage)
    fpsUpdated = Signal(float)
    
    def __init__(self):
        super().__init__()
        self.image_queue = []
        self.queue_mutex = QMutex()
        self.running = True
        
        # FPS计算
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.updateFPS)
        self.fps_timer.start(1000)  # 每秒更新一次FPS
    
    def addImage(self, image: QtGui.QImage):
        """添加图像到处理队列"""
        with QMutexLocker(self.queue_mutex):
            self.image_queue.append(image.copy())
            if len(self.image_queue) > 3:  # 限制队列大小，避免内存堆积
                self.image_queue.pop(0)
    
    def updateFPS(self):
        """更新FPS"""
        current_time = time.time()
        if current_time - self.last_time > 0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.fpsUpdated.emit(self.fps)
            self.frame_count = 0
            self.last_time = current_time
    
    def run(self):
        """线程主循环"""
        while self.running:
            with QMutexLocker(self.queue_mutex):
                if self.image_queue:
                    image = self.image_queue.pop(0)
                    self.frame_count += 1
                    self.imageReady.emit(image)
            
            self.msleep(16)  # ~60 FPS max
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.fps_timer.stop()
        self.wait()


class ImageView(QtWidgets.QGraphicsView):
    """优化的图像显示控件"""
    
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        
        # 设置优化参数
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        
        # 图像相关
        self.current_image = QtGui.QImage()
        self.pixmap_item = None
        self.fps = 0.0
        
        # 创建图像处理线程
        self.image_processor = ImageProcessor()
        self.image_processor.imageReady.connect(self.displayImage)
        self.image_processor.fpsUpdated.connect(self.updateFPS)
        self.image_processor.start()
        
        # 设置场景
        self.setSceneRect(0, 0, 640, 480)
        
    @Slot(QtGui.QImage)
    def displayImage(self, image: QtGui.QImage):
        """在主线程中显示图像"""
        self.current_image = image
        
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        
        pixmap = QtGui.QPixmap.fromImage(image)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        
        # 自动缩放以适应视图
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    @Slot(float)
    def updateFPS(self, fps: float):
        """更新FPS显示"""
        self.fps = fps
        self.scene.invalidate()  # 触发重绘以显示FPS
    
    def updateImage(self, img: QtGui.QImage):
        """接收新图像（从回调函数调用）"""
        self.image_processor.addImage(img)
    
    def saveImage(self):
        """保存当前图像"""
        if not self.current_image.isNull():
            filename = str(Path.home() / "Pictures" / f"clarius_image_{int(time.time())}.png")
            self.current_image.save(filename)
            return filename
        return None
    
    def resizeEvent(self, event):
        """处理窗口大小变化"""
        super().resizeEvent(event)
        w = event.size().width()
        h = event.size().height()
        self.cast.setOutputSize(w, h)
        self.setSceneRect(0, 0, w, h)
        
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
    
    def drawForeground(self, painter, rect):
        """绘制前景（FPS显示）"""
        super().drawForeground(painter, rect)
        
        # 绘制FPS
        painter.setPen(QtGui.QPen(Qt.red))
        font = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)
        painter.setFont(font)
        fps_text = f"FPS: {self.fps:.1f}"
        painter.drawText(rect.topLeft() + QtCore.QPoint(10, 20), fps_text)
    
    def closeEvent(self, event):
        """清理资源"""
        self.image_processor.stop()
        super().closeEvent(event)


class ControlPanel(QtWidgets.QWidget):
    """控制面板"""
    
    def __init__(self, cast, parent=None):
        super().__init__(parent)
        self.cast = cast
        self.setupUI()
    
    def setupUI(self):
        """设置控制面板UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # 连接控制组
        conn_group = QGroupBox("连接控制")
        conn_layout = QGridLayout(conn_group)
        
        self.ip_edit = QtWidgets.QLineEdit("192.168.1.107")
        self.ip_edit.setInputMask("000.000.000.000")
        self.port_edit = QtWidgets.QLineEdit("5828")
        self.port_edit.setInputMask("00000")
        
        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        conn_layout.addWidget(QtWidgets.QLabel("IP地址:"), 0, 0)
        conn_layout.addWidget(self.ip_edit, 0, 1)
        conn_layout.addWidget(QtWidgets.QLabel("端口:"), 1, 0)
        conn_layout.addWidget(self.port_edit, 1, 1)
        conn_layout.addWidget(self.connect_btn, 2, 0, 1, 2)
        
        # 图像控制组
        image_group = QGroupBox("图像控制")
        image_layout = QGridLayout(image_group)
        
        self.freeze_btn = QtWidgets.QPushButton("冻结")
        self.freeze_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        
        image_layout.addWidget(self.freeze_btn, 0, 0, 1, 2)
        
        # 参数控制组
        param_group = QGroupBox("参数控制")
        param_layout = QGridLayout(param_group)
        
        self.depth_dec_btn = QtWidgets.QPushButton("深度 -")
        self.depth_inc_btn = QtWidgets.QPushButton("深度 +")
        self.gain_dec_btn = QtWidgets.QPushButton("增益 -")
        self.gain_inc_btn = QtWidgets.QPushButton("增益 +")
        
        param_layout.addWidget(self.depth_dec_btn, 0, 0)
        param_layout.addWidget(self.depth_inc_btn, 0, 1)
        param_layout.addWidget(self.gain_dec_btn, 1, 0)
        param_layout.addWidget(self.gain_inc_btn, 1, 1)
        
        # 捕获控制组
        capture_group = QGroupBox("捕获控制")
        capture_layout = QGridLayout(capture_group)
        
        self.capture_image_btn = QtWidgets.QPushButton("捕获图像")
        self.capture_cine_btn = QtWidgets.QPushButton("捕获视频")
        self.save_local_btn = QtWidgets.QPushButton("保存本地")
        
        self.capture_image_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        self.capture_cine_btn.setStyleSheet("QPushButton { background-color: #F44336; color: white; }")
        self.save_local_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        
        capture_layout.addWidget(self.capture_image_btn, 0, 0)
        capture_layout.addWidget(self.capture_cine_btn, 0, 1)
        capture_layout.addWidget(self.save_local_btn, 1, 0, 1, 2)
        
        # 模式控制组
        mode_group = QGroupBox("成像模式")
        mode_layout = QGridLayout(mode_group)
        
        self.b_mode_btn = QtWidgets.QPushButton("B模式")
        self.cfi_mode_btn = QtWidgets.QPushButton("彩色模式")
        
        mode_layout.addWidget(self.b_mode_btn, 0, 0)
        mode_layout.addWidget(self.cfi_mode_btn, 0, 1)
        
        # 退出按钮
        self.quit_btn = QtWidgets.QPushButton("退出")
        self.quit_btn.setStyleSheet("QPushButton { background-color: #616161; color: white; font-weight: bold; }")
        
        # 添加到主布局
        layout.addWidget(conn_group)
        layout.addWidget(image_group)
        layout.addWidget(param_group)
        layout.addWidget(capture_group)
        layout.addWidget(mode_group)
        layout.addStretch()
        layout.addWidget(self.quit_btn)


class StatusPanel(QtWidgets.QWidget):
    """状态面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """设置状态面板UI"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # 状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.connection_status = QtWidgets.QLabel("未连接")
        self.connection_status.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        
        self.image_status = QtWidgets.QLabel("图像未运行")
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        
        status_layout.addWidget(QtWidgets.QLabel("连接状态:"))
        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(QtWidgets.QLabel("图像状态:"))
        status_layout.addWidget(self.image_status)
        status_layout.addWidget(self.fps_label)
        
        # 日志显示
        log_group = QGroupBox("操作日志")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(status_group)
        layout.addWidget(log_group)
    
    def updateConnectionStatus(self, connected: bool):
        """更新连接状态"""
        if connected:
            self.connection_status.setText("已连接")
            self.connection_status.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        else:
            self.connection_status.setText("未连接")
            self.connection_status.setStyleSheet("QLabel { color: red; font-weight: bold; }")
    
    def updateImageStatus(self, running: bool):
        """更新图像状态"""
        if running:
            self.image_status.setText("图像运行中")
            self.image_status.setStyleSheet("QLabel { color: green; }")
        else:
            self.image_status.setText("图像已停止")
            self.image_status.setStyleSheet("QLabel { color: orange; }")
    
    def updateFPS(self, fps: float):
        """更新FPS显示"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def addLog(self, message: str):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# 自定义事件类（保持与原代码兼容）
class FreezeEvent(QtCore.QEvent):
    def __init__(self, frozen):
        super().__init__(QtCore.QEvent.User)
        self.frozen = frozen


class ButtonEvent(QtCore.QEvent):
    def __init__(self, btn, clicks):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 1))
        self.btn = btn
        self.clicks = clicks


class ImageEvent(QtCore.QEvent):
    def __init__(self):
        super().__init__(QtCore.QEvent.Type(QtCore.QEvent.User + 2))


class Signaller(QtCore.QObject):
    """信号管理器"""
    freeze = Signal(bool)
    button = Signal(int, int)
    image = Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.usimage = QtGui.QImage()

    def event(self, evt):
        if evt.type() == QtCore.QEvent.User:
            self.freeze.emit(evt.frozen)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 1):
            self.button.emit(evt.btn, evt.clicks)
        elif evt.type() == QtCore.QEvent.Type(QtCore.QEvent.User + 2):
            self.image.emit(self.usimage)
        return True


# 全局信号器
signaller = Signaller()


class MainWindow(QtWidgets.QMainWindow):
    """主窗口"""
    
    def __init__(self, cast):
        super().__init__()
        self.cast = cast
        self.setupUI()
        self.connectSignals()
        self.initializeCast()
    
    def setupUI(self):
        """设置主窗口UI"""
        self.setWindowTitle("Clarius Cast Demo")
        self.setMinimumSize(1000, 700)
        
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 创建图像视图
        self.image_view = ImageView(self.cast)
        
        # 创建右侧面板
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # 创建控制面板和状态面板
        self.control_panel = ControlPanel(self.cast)
        self.status_panel = StatusPanel()
        
        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(self.status_panel)
        
        # 设置分割器
        main_splitter.addWidget(self.image_view)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([700, 300])  # 设置初始比例
        
        # 设置主布局
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        
        # 创建状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def connectSignals(self):
        """连接信号和槽"""
        # 控制面板信号
        self.control_panel.connect_btn.clicked.connect(self.toggleConnection)
        self.control_panel.freeze_btn.clicked.connect(self.toggleFreeze)
        self.control_panel.depth_dec_btn.clicked.connect(lambda: self.sendCommand(CMD_DEPTH_DEC))
        self.control_panel.depth_inc_btn.clicked.connect(lambda: self.sendCommand(CMD_DEPTH_INC))
        self.control_panel.gain_dec_btn.clicked.connect(lambda: self.sendCommand(CMD_GAIN_DEC))
        self.control_panel.gain_inc_btn.clicked.connect(lambda: self.sendCommand(CMD_GAIN_INC))
        self.control_panel.capture_image_btn.clicked.connect(lambda: self.sendCommand(CMD_CAPTURE_IMAGE))
        self.control_panel.capture_cine_btn.clicked.connect(lambda: self.sendCommand(CMD_CAPTURE_CINE))
        self.control_panel.save_local_btn.clicked.connect(self.saveLocalImage)
        self.control_panel.b_mode_btn.clicked.connect(lambda: self.sendCommand(CMD_B_MODE))
        self.control_panel.cfi_mode_btn.clicked.connect(lambda: self.sendCommand(CMD_CFI_MODE))
        self.control_panel.quit_btn.clicked.connect(self.shutdown)
        
        # Cast信号
        signaller.freeze.connect(self.onFreezeChanged)
        signaller.button.connect(self.onButtonPressed)
        signaller.image.connect(self.onNewImage)
        
        # 图像处理器FPS信号
        self.image_view.image_processor.fpsUpdated.connect(self.status_panel.updateFPS)
    
    def initializeCast(self):
        """初始化Cast"""
        path = os.path.expanduser("~/")
        if self.cast.init(path, 640, 480):
            self.status_panel.addLog("Cast初始化成功")
        else:
            self.status_panel.addLog("Cast初始化失败")
    
    @Slot()
    def toggleConnection(self):
        """切换连接状态"""
        if not self.cast.isConnected():
            ip = self.control_panel.ip_edit.text()
            port = int(self.control_panel.port_edit.text())
            
            if self.cast.connect(ip, port, "research"):
                self.control_panel.connect_btn.setText("断开连接")
                self.control_panel.connect_btn.setStyleSheet("QPushButton { background-color: #F44336; color: white; font-weight: bold; }")
                self.status_panel.updateConnectionStatus(True)
                self.status_panel.addLog(f"已连接到 {ip}:{port}")
            else:
                self.status_panel.addLog(f"连接失败: {ip}:{port}")
        else:
            if self.cast.disconnect():
                self.control_panel.connect_btn.setText("连接")
                self.control_panel.connect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
                self.status_panel.updateConnectionStatus(False)
                self.status_panel.addLog("已断开连接")
            else:
                self.status_panel.addLog("断开连接失败")
    
    @Slot()
    def toggleFreeze(self):
        """切换冻结状态"""
        if self.cast.isConnected():
            self.sendCommand(CMD_FREEZE)
    
    def sendCommand(self, command):
        """发送命令"""
        if self.cast.isConnected():
            self.cast.userFunction(command, 0)
            command_names = {
                CMD_FREEZE: "冻结/解冻",
                CMD_DEPTH_DEC: "减少深度",
                CMD_DEPTH_INC: "增加深度",
                CMD_GAIN_DEC: "减少增益",
                CMD_GAIN_INC: "增加增益",
                CMD_CAPTURE_IMAGE: "捕获图像",
                CMD_CAPTURE_CINE: "捕获视频",
                CMD_B_MODE: "B模式",
                CMD_CFI_MODE: "彩色模式"
            }
            self.status_panel.addLog(f"执行命令: {command_names.get(command, f'命令{command}')}")
    
    @Slot()
    def saveLocalImage(self):
        """保存本地图像"""
        filename = self.image_view.saveImage()
        if filename:
            self.status_panel.addLog(f"图像已保存: {filename}")
        else:
            self.status_panel.addLog("保存失败: 没有可用图像")
    
    @Slot(bool)
    def onFreezeChanged(self, frozen):
        """处理冻结状态变化"""
        if frozen:
            self.control_panel.freeze_btn.setText("运行")
            self.control_panel.freeze_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            self.status_panel.updateImageStatus(False)
        else:
            self.control_panel.freeze_btn.setText("冻结")
            self.control_panel.freeze_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
            self.status_panel.updateImageStatus(True)
    
    @Slot(int, int)
    def onButtonPressed(self, btn, clicks):
        """处理按钮按下事件"""
        self.status_panel.addLog(f"按钮 {btn} 被按下 {clicks} 次")
    
    @Slot(QtGui.QImage)
    def onNewImage(self, img):
        """处理新图像"""
        self.image_view.updateImage(img)
    
    @Slot()
    def shutdown(self):
        """关闭应用"""
        # 停止图像处理线程
        self.image_view.image_processor.stop()
        
        # 清理Cast资源
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
        self.cast.destroy()
        
        # 退出应用
        QtWidgets.QApplication.quit()


# 回调函数（保持与原代码兼容）
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    """新处理图像回调"""
    bpp = sz / (width * height)
    if bpp == 4:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
    else:
        img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
    
    signaller.usimage = img.copy()
    evt = ImageEvent()
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    """新原始图像回调"""
    return


def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    """新频谱图像回调"""
    return


def newImuData(imu):
    """新IMU数据回调"""
    return


def freezeFn(frozen):
    """冻结状态变化回调"""
    evt = FreezeEvent(frozen)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


def buttonsFn(button, clicks):
    """按钮按下回调"""
    evt = ButtonEvent(button, clicks)
    QtCore.QCoreApplication.postEvent(signaller, evt)
    return


def main():
    """主函数"""
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建Cast对象
    cast = pyclariuscast.Caster(
        newProcessedImage, newRawImage, newSpectrumImage, 
        newImuData, freezeFn, buttonsFn
    )
    
    # 创建主窗口
    window = MainWindow(cast)
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()