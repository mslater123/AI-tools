import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image


class VideoSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Segmentation")
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(800, 600)  # Set fixed size for the window

        # Initialize Segformer model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(self.device)

        self.video_path = None
        self.export_dir = None

        # Setup UI
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

        self.export_button = QPushButton("Export Segmented Frames", self)
        self.export_button.clicked.connect(self.export_frames)
        layout.addWidget(self.export_button)

        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.segment_label = QLabel(self)
        layout.addWidget(self.segment_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            if ret:
                segmented_frame = self.segment_frame(frame)
                self.display_frame(frame, self.video_label)
                self.display_frame(segmented_frame, self.segment_label)
            self.cap.release()

    def display_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def segmentation_to_color(self, seg_map):
        color_segmented_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for class_id in np.unique(seg_map):
            color_segmented_image[seg_map == class_id] = self.class_id_to_color(class_id)
        return color_segmented_image

    def class_id_to_color(self, class_id):
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 256, 3))

    def segment_frame(self, frame):
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        color_segmented_image = self.segmentation_to_color(pred_seg)
        return color_segmented_image

    def export_frames(self):
        if self.video_path:
            self.export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if self.export_dir:
                self.cap = cv2.VideoCapture(self.video_path)
                frame_count = 0
                ret, frame = self.cap.read()
                while ret:
                    segmented_frame = self.segment_frame(frame)
                    self.display_frame(segmented_frame, self.segment_label)
                    export_path = os.path.join(self.export_dir, f"seg_{frame_count:06d}.png")
                    cv2.imwrite(export_path, segmented_frame)
                    frame_count += 1
                    ret, frame = self.cap.read()
                self.cap.release()
                print(f"Segmented frames saved to {self.export_dir}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSegmentationApp()
    window.show()
    sys.exit(app.exec())
