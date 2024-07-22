import sys
import subprocess
import re
import time
import torch
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer


def get_gpu_memory_usage_nvidia_smi():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader']
        )
        result = result.decode('utf-8').strip().split('\n')

        gpu_memory_info = []
        for i, line in enumerate(result):
            total_memory, used_memory, free_memory = map(str.strip, line.split(','))
            total_memory = int(total_memory)
            used_memory = int(used_memory)
            free_memory = int(free_memory)
            gpu_memory_info.append((i, total_memory, used_memory, free_memory))

        return gpu_memory_info
    except subprocess.CalledProcessError as e:
        print(f"Failed to run nvidia-smi: {e}")
        return []
    except ValueError as e:
        print(f"Failed to parse GPU memory values: {e}")
        return []


class GPUMonitorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GPU Memory Monitor')
        self.layout = QVBoxLayout()
        self.labels = []

        # Initialize labels for each GPU
        self.update_labels()

        self.setLayout(self.layout)

        # Set up a timer to update GPU memory usage every 5 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_labels)
        self.timer.start(5000)  # 5000 milliseconds = 5 seconds

    def update_labels(self):
        gpu_memory_info = get_gpu_memory_usage_nvidia_smi()

        # Remove old labels
        for label in self.labels:
            self.layout.removeWidget(label)
            label.deleteLater()

        self.labels = []

        for i, (gpu_id, total_memory, used_memory, free_memory) in enumerate(gpu_memory_info):
            label = QLabel(
                f"GPU {gpu_id}:\n  Total GPU Memory: {total_memory / 1024:.2f} GB\n  Used GPU Memory: {used_memory / 1024:.2f} GB\n  Free GPU Memory: {free_memory / 1024:.2f} GB")
            self.labels.append(label)
            self.layout.addWidget(label)


def main():
    app = QApplication(sys.argv)
    window = GPUMonitorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
