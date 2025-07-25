import ctypes
import os.path
import sys
from pathlib import Path
from typing import Final

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot

import pyclariuscast

CMD_FREEZE: Final = 1
CMD_CAPTURE_IMAGE: Final = 2
CMD_CAPTURE_CINE: Final = 3
CMD_DEPTH_DEC: Final = 4
CMD_DEPTH_INC: Final = 5
CMD_GAIN_DEC: Final = 6
CMD_GAIN_INC: Final = 7
CMD_B_MODE: Final = 12
CMD_CFI_MODE: Final = 14


class USThread(QtCore.QThread):
    image_ready = Signal(QtGui.QImage)
    freeze_changed = Signal(bool)
    button_pressed = Signal(int, int)
    caster_ready = Signal()  # ✅ 通知主线程 caster 可用

    def __init__(self, parent=None):
        super().__init__(parent)
        self._caster = None
        self._running = False

    def run(self):
        print("[USThread] Starting run()...")
        self._caster = pyclariuscast.Caster(
            self.newProcessedImage,
            self.newRawImage,
            self.newSpectrumImage,
            self.newImuData,
            self.freezeFn,
            self.buttonsFn
        )

        if not self._caster.init(str(Path.home()), 640, 480):
            print("Caster failed to initialize")
            return

        self.caster_ready.emit()
        self._running = True
        while self._running:
            self.msleep(10)

    def stop(self):
        self._running = False
        if self._caster:
            self._caster.destroy()
        self.quit()
        self.wait()

    def newProcessedImage(self, image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
        bpp = sz / (width * height)
        if bpp == 4:
            img = QtGui.QImage(image, width, height, QtGui.QImage.Format_ARGB32)
        else:
            img = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
        self.image_ready.emit(img.copy())  # ✅ copy 保证图像安全

    def newRawImage(self, *args): pass
    def newSpectrumImage(self, *args): pass
    def newImuData(self, *args): pass

    def freezeFn(self, frozen): self.freeze_changed.emit(frozen)
    def buttonsFn(self, button, clicks): self.button_pressed.emit(button, clicks)

    def connect_device(self, ip, port, protocol="research"):
        return self._caster.connect(ip, port, protocol) if self._caster else False

    def disconnect_device(self):
        return self._caster.disconnect() if self._caster else False

    def isConnected(self):
        return self._caster.isConnected() if self._caster else False

    def setOutputSize(self, w, h):
        if self._caster:
            self._caster.setOutputSize(w, h)

    def userFunction(self, cmd, val):
        if self.isConnected():
            self._caster.userFunction(cmd, val)

    # wrapped commands
    def tryFreeze(self): self.userFunction(CMD_FREEZE, 0)
    def tryDepthUp(self): self.userFunction(CMD_DEPTH_INC, 0)
    def tryDepthDown(self): self.userFunction(CMD_DEPTH_DEC, 0)
    def tryGainDec(self): self.userFunction(CMD_GAIN_DEC, 0)
    def tryGainInc(self): self.userFunction(CMD_GAIN_INC, 0)
    def tryCaptureImage(self): self.userFunction(CMD_CAPTURE_IMAGE, 0)
    def tryCaptureCine(self): self.userFunction(CMD_CAPTURE_CINE, 0)
    def tryBMode(self): self.userFunction(CMD_B_MODE, 0)
    def tryCfiMode(self): self.userFunction(CMD_CFI_MODE, 0)


class USImageView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene())
        self.image = QtGui.QImage()

    def updateImage(self, img: QtGui.QImage):
        self.image = img
        self.scene().invalidate()

    def resizeEvent(self, evt):
        w, h = evt.size().width(), evt.size().height()
        self.setSceneRect(0, 0, w, h)

    def drawForeground(self, painter, rect):
        if not self.image.isNull():
            painter.drawImage(rect, self.image)


class USWidget(QtWidgets.QMainWindow):
    def __init__(self, usthread: USThread, parent=None):
        super().__init__(parent)
        self.usthread = usthread
        self.setWindowTitle("Ultrasound Viewer")

        self.image_view = USImageView()
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.run_button = QtWidgets.QPushButton("Run")
        self.quit_button = QtWidgets.QPushButton("Quit")
        self.capture_image_button = QtWidgets.QPushButton("Capture Image")
        self.capture_cine_button = QtWidgets.QPushButton("Capture Cine")
        self.depth_up_button = QtWidgets.QPushButton("Depth +")
        self.depth_down_button = QtWidgets.QPushButton("Depth -")
        self.gain_up_button = QtWidgets.QPushButton("Gain +")
        self.gain_down_button = QtWidgets.QPushButton("Gain -")
        self.b_mode_button = QtWidgets.QPushButton("B Mode")
        self.cfi_mode_button = QtWidgets.QPushButton("CFI Mode")

        self.ip_input = QtWidgets.QLineEdit("192.168.1.108")
        self.port_input = QtWidgets.QLineEdit("5828")
        self.connect_button.setEnabled(False)  # 等待线程初始化

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_view)

        conn_layout = QtWidgets.QHBoxLayout()
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.connect_button)
        conn_layout.addWidget(self.run_button)
        conn_layout.addWidget(self.quit_button)
        layout.addLayout(conn_layout)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(self.depth_up_button)
        control_layout.addWidget(self.depth_down_button)
        control_layout.addWidget(self.gain_up_button)
        control_layout.addWidget(self.gain_down_button)
        layout.addLayout(control_layout)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(self.b_mode_button)
        mode_layout.addWidget(self.cfi_mode_button)
        layout.addLayout(mode_layout)

        capture_layout = QtWidgets.QHBoxLayout()
        capture_layout.addWidget(self.capture_image_button)
        capture_layout.addWidget(self.capture_cine_button)
        layout.addLayout(capture_layout)

        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # slot connections
        self.connect_button.clicked.connect(self.try_connect)
        self.run_button.clicked.connect(self.usthread.tryFreeze)
        self.quit_button.clicked.connect(self.shutdown)
        self.capture_image_button.clicked.connect(self.usthread.tryCaptureImage)
        self.capture_cine_button.clicked.connect(self.usthread.tryCaptureCine)
        self.depth_up_button.clicked.connect(self.usthread.tryDepthUp)
        self.depth_down_button.clicked.connect(self.usthread.tryDepthDown)
        self.gain_up_button.clicked.connect(self.usthread.tryGainInc)
        self.gain_down_button.clicked.connect(self.usthread.tryGainDec)
        self.b_mode_button.clicked.connect(self.usthread.tryBMode)
        self.cfi_mode_button.clicked.connect(self.usthread.tryCfiMode)

        # signal slot
        self.usthread.image_ready.connect(self.onImageReady)
        self.usthread.freeze_changed.connect(self.onFreeze)
        self.usthread.button_pressed.connect(self.onButton)
        self.usthread.caster_ready.connect(self.onCasterReady)

    @Slot()
    def onCasterReady(self):
        self.statusBar().showMessage("Caster Initialized")
        self.connect_button.setEnabled(True)

    @Slot(QtGui.QImage)
    def onImageReady(self, img):
        self.image_view.updateImage(img)

    @Slot(bool)
    def onFreeze(self, frozen):
        self.run_button.setText("Run" if frozen else "Freeze")

    @Slot(int, int)
    def onButton(self, btn, clicks):
        self.statusBar().showMessage(f"Button {btn} clicked {clicks} times")

    def try_connect(self):
        ip = self.ip_input.text()
        port = int(self.port_input.text())
        if not self.usthread.isConnected():
            if self.usthread.connect_device(ip, port):
                self.statusBar().showMessage("Connected")
            else:
                self.statusBar().showMessage("Connection Failed")
        else:
            if self.usthread.disconnect_device():
                self.statusBar().showMessage("Disconnected")

    def shutdown(self):
        self.usthread.stop()
        QtWidgets.QApplication.quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    usthread = USThread()
    usthread.start()
    window = USWidget(usthread)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
