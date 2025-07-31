import cv2
import numpy as np
import torch
import sys
import os
import queue
from pygrabber.dshow_graph import FilterGraph
from PIL import Image
from ml.inferencer import UNetInferencer
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QSlider, QCheckBox
from PySide6.QtGui import QImage, QPixmap, QIcon, QPainter
from PySide6.QtSvg import QSvgRenderer

class VideoRecorder:
    def __init__(self, device_name="A325"):
        self.device_name = device_name

    def init_capture_device(self) -> cv2.VideoCapture:
        # get the video capture device
        graph = FilterGraph()
        devices = graph.get_input_devices()
        target = next((i for i,n in enumerate(devices) if self.device_name in n), None)
        if target is None:
            raise RuntimeError(f"Failed to find device containing '{self.device_name}'. Please check the connection.")

        print(f"Using device: {devices[target]}")

        # Initialize the video capture
        cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        desired_w, desired_h = 2388/2, 1668/2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  desired_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open device {target}.")
        return cap


class VideoThread(QThread):
    frame_signal = Signal(np.ndarray)
    
    def __init__(self, recorder, frame_queue):
        super().__init__()
        self.recorder = recorder
        self.frame_queue = frame_queue  # 添加队列用于与MaskThread通信
        self.cap = None
        self.running = False
        self.crop_params = {
            'crop_top': 110,
            'crop_bottom': 130,
            'crop_left': 400,
            'crop_right': 400
        }
        
    def run(self):
        self.cap = self.recorder.init_capture_device()
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            # Apply cropping
            cropped_frame = self.crop_image(frame, **self.crop_params)
            
            # 将裁剪后的帧放入队列，而不是直接发送给GUI
            try:
                self.frame_queue.put(cropped_frame, block=False)
            except queue.Full:
                # 如果队列满了，跳过这一帧以避免阻塞
                pass
            
            self.msleep(33)  # ~30 FPS
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
    
    def update_crop_params(self, crop_params):
        self.crop_params = crop_params
    
    def crop_image(self, img: np.ndarray, crop_top: int = 0, crop_bottom: int = 0, 
                   crop_left: int = 0, crop_right: int = 0) -> np.ndarray:
        """Crop the given image by removing specified pixels from each border."""
        h, w = img.shape[:2]
        
        # Ensure cropping isn't too large
        if crop_top + crop_bottom >= h:
            crop_top = crop_bottom = 0
        if crop_left + crop_right >= w:
            crop_left = crop_right = 0
            
        return img[crop_top:h - crop_bottom, crop_left:w - crop_right]

class MaskThread(QThread):
    processed_frame_signal = Signal(np.ndarray)  # 发送处理后的帧给GUI
    
    def __init__(self, inferencer, frame_queue):
        super().__init__()
        self.inferencer = inferencer
        self.frame_queue = frame_queue
        self.running = False
        self.enable_inference = False
        self.alpha = 0.3  # transparency for overlay
        
    def run(self):
        self.running = True
        
        while self.running:
            try:
                # Get cropped frame from queue
                cropped_frame = self.frame_queue.get(timeout=0.1)
                
                if self.enable_inference:
                    # Execute inference and overlay mask
                    processed_frame = self.process_inference(cropped_frame)
                else:
                    # Directly use the original frame
                    processed_frame = cropped_frame

                # Send processed frame to GUI
                self.processed_frame_signal.emit(processed_frame)
                
            except queue.Empty:
                # If the queue is empty, continue the loop
                continue
            except Exception as e:
                print(f"MaskThread error: {e}")
                continue
                
    def stop(self):
        self.running = False
        self.wait()
        
    def set_inference_enabled(self, enabled):
        """Set whether to enable inference"""
        self.enable_inference = enabled

    def set_alpha(self, alpha):
        """Set transparency"""
        self.alpha = alpha
        
    def process_inference(self, frame):
        """Process frame with UNet inference and overlay mask"""
        try:
            # Convert to grayscale for inference
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # To np array
            img_np = np.array(frame_gray)
            
            # Run inference
            inference_result = self.inferencer.predict_array(img_np)
            
            # Create overlay
            inference_display = frame.copy()
            
            # Ensure mask and image dimensions match
            if inference_result.shape[:2] != frame.shape[:2]:
                mask_pil = Image.fromarray(inference_result)
                mask_pil = mask_pil.resize((frame.shape[1], frame.shape[0]), Image.Resampling.LANCZOS)
                inference_result = np.array(mask_pil)
            
            # Apply green overlay
            mask_active = inference_result > 127
            active_pixels = np.sum(mask_active)
            
            if active_pixels > 0:
                inference_display[mask_active, 0] = inference_display[mask_active, 0] * (1 - self.alpha)  # B
                inference_display[mask_active, 1] = inference_display[mask_active, 1] * (1 - self.alpha) + 255 * self.alpha  # G
                inference_display[mask_active, 2] = inference_display[mask_active, 2] * (1 - self.alpha)  # R
            
            return inference_display
            
        except Exception as e:
            print(f"Inference processing error: {e}")
            return frame

class VideoRecorderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Video Recording with UNet Inference")
        self.setGeometry(100, 100, 1400, 800)

        # load application icon
        self.load_app_icon()
        
        # load application stylesheet
        self.load_stylesheet()
        
        # Initialize components
        self.recorder = VideoRecorder()
        
        # create frame queue for communication between VideoThread and MaskThread
        self.frame_queue = queue.Queue(maxsize=20)

        # Initialize inferencer
        try:
            self.inferencer = UNetInferencer(ckpt_path="ml/checkpoints/best_model_v1.pth")
            print("UNet inferencer initialized successfully")
        except Exception as e:
            print(f"UNet inferencer initialization failed: {e}")
            self.inferencer = None

        # create threads for video capture and inference
        self.video_thread = VideoThread(self.recorder, self.frame_queue)
        self.mask_thread = MaskThread(self.inferencer, self.frame_queue) if self.inferencer else None
        
        # connect signals
        if self.mask_thread:
            self.mask_thread.processed_frame_signal.connect(self.update_frame)
        
        # Variables
        self.original_frame = None
        self.enable_inference = False
        self.alpha = 0.3  # transparency for overlay
        
        # UI setup
        self.setup_ui()
        
    def load_app_icon(self):
        """Load application icon"""
        try:
            # Get current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(current_dir, "supportfiles", "icon_binh.svg")
            
            if os.path.exists(icon_path):
                # Create icon using SVG renderer
                svg_renderer = QSvgRenderer(icon_path)
                if svg_renderer.isValid():
                    # Create a QPixmap to render SVG
                    pixmap = QPixmap(64, 64)  # Set icon size to 64x64
                    pixmap.fill(Qt.transparent)

                    # Use QPainter to render SVG to QPixmap
                    painter = QPainter(pixmap)
                    svg_renderer.render(painter)
                    painter.end()

                    # Create QIcon
                    icon = QIcon(pixmap)
                    
                    # Set window icon (shows in window title bar)
                    self.setWindowIcon(icon)

                    # Set application icon (shows in Windows taskbar and desktop)
                    QtWidgets.QApplication.instance().setWindowIcon(icon)
                    
                    print(f"Successfully loaded application icon: {icon_path}")
                else:
                    print(f"Invalid SVG file: {icon_path}")
            else:
                print(f"Icon file does not exist: {icon_path}")
                # Use system default icon as fallback
                default_icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
                self.setWindowIcon(default_icon)
                QtWidgets.QApplication.instance().setWindowIcon(default_icon)
        except Exception as e:
            print(f"Error loading icon: {e}")
            # Use system default icon as fallback
            default_icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
            self.setWindowIcon(default_icon)
            QtWidgets.QApplication.instance().setWindowIcon(default_icon)
    
    def load_stylesheet(self):
        """Load application stylesheet"""
        try:
            # Get current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            style_path = os.path.join(current_dir, "supportfiles", "style.qss")
            
            if os.path.exists(style_path):
                with open(style_path, 'r', encoding='utf-8') as f:
                    style_sheet = f.read()
                    self.setStyleSheet(style_sheet)
                    print(f"Successfully loaded stylesheet: {style_path}")
            else:
                print(f"Stylesheet file does not exist: {style_path}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for video display
        video_layout = QVBoxLayout()
        
        # Single video display label
        self.video_label = QLabel("Video Display")
        self.video_label.setMinimumSize(800, 600) 
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # Status label to show current mode
        self.status_label = QLabel("Ready - Click 'Start Video' to begin")
        self.status_label.setStyleSheet("font-size: 14px; color: blue; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        video_layout.addWidget(self.status_label)
        video_layout.addWidget(self.video_label)
        
        # Right panel for controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setMaximumWidth(300)
        
        # Control buttons
        self.start_button = QPushButton("Start Video")
        self.start_button.clicked.connect(self.start_video)
        
        self.stop_button = QPushButton("Stop Video")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        
        # Inference toggle
        self.inference_checkbox = QCheckBox("Enable Real-time Inference")
        self.inference_checkbox.setChecked(False)  # Default is off

        # Connect signals
        self.inference_checkbox.stateChanged.connect(lambda state: self.toggle_inference(state))
        
        # Transparency slider
        transparency_layout = QVBoxLayout()
        transparency_layout.addWidget(QLabel("Overlay Transparency:"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(int(self.alpha * 100))
        self.transparency_slider.valueChanged.connect(self.update_transparency)
        self.transparency_label = QLabel(f"{int(self.alpha * 100)}%")
        transparency_layout.addWidget(self.transparency_slider)
        transparency_layout.addWidget(self.transparency_label)
        
        # Crop parameter sliders
        self.setup_crop_controls(control_layout)
        
        # Add controls to layout
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.inference_checkbox)
        control_layout.addLayout(transparency_layout)
        control_layout.addStretch()
        
        # Add to main layout
        main_layout.addLayout(video_layout, 3)
        main_layout.addWidget(control_widget, 1)
        
    def setup_crop_controls(self, layout):
        """Setup crop parameter controls"""
        crop_group = QtWidgets.QGroupBox("Crop Parameters")
        crop_layout = QVBoxLayout(crop_group)
        
        self.crop_sliders = {}
        crop_params = [
            ('crop_top', 'Top', 0, 500, 110),
            ('crop_bottom', 'Bottom', 0, 500, 130),
            ('crop_left', 'Left', 0, 800, 400),
            ('crop_right', 'Right', 0, 800, 400)
        ]
        
        for param, name, min_val, max_val, default in crop_params:
            param_layout = QVBoxLayout()
            param_layout.addWidget(QLabel(f"{name}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.valueChanged.connect(self.update_crop_params)
            
            label = QLabel(str(default))
            
            param_layout.addWidget(slider)
            param_layout.addWidget(label)
            
            self.crop_sliders[param] = (slider, label)
            crop_layout.addLayout(param_layout)
            
        layout.addWidget(crop_group)
        
    def update_crop_params(self):
        """Update crop parameters from sliders"""
        crop_params = {}
        for param, (slider, label) in self.crop_sliders.items():
            value = slider.value()
            label.setText(str(value))
            crop_params[param] = value
            
        if hasattr(self, 'video_thread'):
            self.video_thread.update_crop_params(crop_params)
    
    def start_video(self):
        """Start video capture and display"""
        if self.mask_thread:
            self.mask_thread.start()
        self.video_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
    def stop_video(self):
        """Stop video capture"""
        self.video_thread.stop()
        if self.mask_thread:
            self.mask_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Clear display
        self.video_label.clear()
        self.video_label.setText("Video Display")
        self.status_label.setText("Ready - Click 'Start Video' to begin")
        
    def toggle_inference(self, state):
        """Toggle real-time inference"""
        if not self.mask_thread:
            QtWidgets.QMessageBox.warning(
                self, "inference error", 
                "UNet inferencer initialization failed, unable to enable inference."
            )
            self.inference_checkbox.setChecked(False)
            return
        
        print(f"Real-time inference {'enabled' if state else 'disabled'}")
        if state == 2:  # Qt.Checked is 2
            print("start...")
            self.enable_inference = True
            self.mask_thread.set_inference_enabled(True)
            self.status_label.setText("Real-time Inference with Green Overlay")
            print("Real-time inference enabled")
        else:
            print("Stopping real-time inference...")
            self.enable_inference = False
            self.mask_thread.set_inference_enabled(False)
            self.status_label.setText("Original Video - Raw Stream")
            print("Real-time inference disabled")

    def update_transparency(self, value):
        """Update overlay transparency"""
        self.alpha = value / 100.0
        self.transparency_label.setText(f"{value}%")
        if self.mask_thread:
            self.mask_thread.set_alpha(self.alpha)
        
    def update_frame(self, frame):
        """Update video frame in display - received from MaskThread"""
        self.original_frame = frame.copy()
        
        # directly display the processed frame from MaskThread
        qimg = self.cv2_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def cv2_to_qimage(self, cv_img):
        """Convert OpenCV image to QImage"""
        # Ensure the image is contiguous in memory
        if not cv_img.flags['C_CONTIGUOUS']:
            cv_img = np.ascontiguousarray(cv_img)
            
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_image.rgbSwapped()  # BGR to RGB
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.video_thread.isRunning():
            self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application properties for Windows taskbar/desktop
    app.setApplicationName("Real-time Video Recording with UNet Inference")
    app.setApplicationDisplayName("UNet Video Recorder")
    app.setOrganizationName("Clarius Research")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = VideoRecorderGUI()
    window.show()
    
    sys.exit(app.exec())
