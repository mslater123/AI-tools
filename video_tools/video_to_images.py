import sys
import cv2
import os
import traceback
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar)
from PyQt6.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
import threading


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)


class FrameExtractor(QRunnable):
    def __init__(self, video_path, output_folder, frame_indices, signals):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_indices = frame_indices
        self.signals = signals

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception(f"Error: Unable to open video file {self.video_path}")

            for frame_index in self.frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_filename = os.path.join(self.output_folder, f"frame_{frame_index:05d}.png")
                cv2.imwrite(frame_filename, frame)
                self.signals.progress.emit(1)

            cap.release()
        except Exception as e:
            self.signals.error.emit(str(e))
        self.signals.finished.emit()


class VideoToFramesConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video to Frames Converter')
        self.resize(400, 200)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.video_path_label = QLabel('No video file selected')
        self.layout.addWidget(self.video_path_label)

        self.select_video_button = QPushButton('Select Video File')
        self.select_video_button.clicked.connect(self.select_video_file)
        self.layout.addWidget(self.select_video_button)

        self.output_folder_label = QLabel('No output folder selected')
        self.layout.addWidget(self.output_folder_label)

        self.select_output_folder_button = QPushButton('Select Output Folder')
        self.select_output_folder_button.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.select_output_folder_button)

        self.start_button = QPushButton('Start Conversion')
        self.start_button.clicked.connect(self.start_conversion)
        self.layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.video_path = None
        self.output_folder = None
        self.thread_pool = QThreadPool()

    def select_video_file(self):
        self.video_path = QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.mp4 *.avi *.mov)')[
            0]
        self.video_path_label.setText(self.video_path if self.video_path else 'No video file selected')

    def select_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        self.output_folder_label.setText(self.output_folder if self.output_folder else 'No output folder selected')

    def start_conversion(self):
        if not self.video_path or not self.output_folder:
            print("Please select a video file and an output folder")
            return

        self.progress_bar.setValue(0)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        num_threads = 8
        frames_per_thread = total_frames // num_threads

        self.progress_bar.setMaximum(total_frames)
        self.total_progress = 0

        for i in range(num_threads):
            start_frame = i * frames_per_thread
            if i == num_threads - 1:
                end_frame = total_frames
            else:
                end_frame = (i + 1) * frames_per_thread

            frame_indices = list(range(start_frame, end_frame))

            signals = WorkerSignals()
            signals.progress.connect(self.update_progress)
            signals.finished.connect(self.worker_finished)
            signals.error.connect(self.worker_error)

            worker = FrameExtractor(self.video_path, self.output_folder, frame_indices, signals)
            self.thread_pool.start(worker)

    def update_progress(self, value):
        self.total_progress += value
        self.progress_bar.setValue(self.total_progress)

    def worker_finished(self):
        if self.total_progress >= self.progress_bar.maximum():
            self.progress_bar.setValue(self.progress_bar.maximum())
            print("Conversion finished")

    def worker_error(self, error_message):
        print(f"Error: {error_message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoToFramesConverter()
    window.show()
    sys.exit(app.exec())
