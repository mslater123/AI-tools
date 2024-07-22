import os
import sys
import cv2
import traceback
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QLineEdit, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import normalize
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, DPTImageProcessor, DPTForDepthEstimation, AutoModelForImageSegmentation, SamModel, SamProcessor
import torch.nn.functional as F
from skimage import io

CUDA_LAUNCH_BLOCKING=1

class ImageSlider(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Slider')
        self.resize(1500, 700)
        self.image_files = []
        self.current_index = 0

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton('Load Image Directory')
        self.load_button.clicked.connect(self.load_directory)
        self.layout.addWidget(self.load_button)

        self.image_layout = QHBoxLayout()

        self.init_original_image_layout()
        self.init_segmented_image_layout()
        self.init_depth_image_layout()
        self.init_multiplied_image_layout()
        self.init_no_bg_image_layout()
        self.init_sam_image_layout()
        self.init_false_color_image_layout()

        self.layout.addLayout(self.image_layout)

        self.nav_layout = QHBoxLayout()

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.go_back)
        self.nav_layout.addWidget(self.back_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.nav_layout.addWidget(self.slider)

        self.forward_button = QPushButton('Forward')
        self.forward_button.clicked.connect(self.go_forward)
        self.nav_layout.addWidget(self.forward_button)

        self.layout.addLayout(self.nav_layout)

        self.export_button = QPushButton('Export Images')
        self.export_button.clicked.connect(self.export_images)
        self.layout.addWidget(self.export_button)

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

            self.seg_processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(self.device)

            self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)

            self.no_bg_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(self.device)

            self.sam_processor = SamProcessor.from_pretrained("nielsr/slimsam-50-uniform")
            self.sam_model = SamModel.from_pretrained("nielsr/slimsam-50-uniform").to(self.device1)

        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()

    def init_original_image_layout(self):
        self.original_image_layout = QVBoxLayout()

        self.original_export_layout = QHBoxLayout()
        self.original_export_path = QLineEdit(self)
        self.original_export_path.setPlaceholderText("Original Image Export Path")
        self.original_export_layout.addWidget(self.original_export_path)
        self.original_export_checkbox = QCheckBox("Export")
        self.original_export_checkbox.setChecked(True)
        self.original_export_layout.addWidget(self.original_export_checkbox)
        self.original_image_layout.addLayout(self.original_export_layout)

        self.original_image_label = QLabel('No Image Loaded')
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_layout.addWidget(self.original_image_label)
        self.original_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.original_exposure_slider.setMinimum(-100)
        self.original_exposure_slider.setMaximum(100)
        self.original_exposure_slider.setValue(0)
        self.original_exposure_slider.valueChanged.connect(self.update_original_image)
        self.original_image_layout.addWidget(self.original_exposure_slider)

        self.original_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.original_contrast_slider.setMinimum(-100)
        self.original_contrast_slider.setMaximum(100)
        self.original_contrast_slider.setValue(0)
        self.original_contrast_slider.valueChanged.connect(self.update_original_image)
        self.original_image_layout.addWidget(self.original_contrast_slider)

        self.image_layout.addLayout(self.original_image_layout)

    def init_segmented_image_layout(self):
        self.segmented_image_layout = QVBoxLayout()

        self.segmented_export_layout = QHBoxLayout()
        self.segmented_export_path = QLineEdit(self)
        self.segmented_export_path.setPlaceholderText("Segmented Image Export Path")
        self.segmented_export_layout.addWidget(self.segmented_export_path)
        self.segmented_export_checkbox = QCheckBox("Export")
        self.segmented_export_checkbox.setChecked(True)
        self.segmented_export_layout.addWidget(self.segmented_export_checkbox)
        self.segmented_image_layout.addLayout(self.segmented_export_layout)

        self.segmented_image_label = QLabel('No Segmented Image')
        self.segmented_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image_layout.addWidget(self.segmented_image_label)
        self.segmented_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.segmented_exposure_slider.setMinimum(-100)
        self.segmented_exposure_slider.setMaximum(100)
        self.segmented_exposure_slider.setValue(0)
        self.segmented_exposure_slider.valueChanged.connect(self.update_segmented_image)
        self.segmented_image_layout.addWidget(self.segmented_exposure_slider)

        self.segmented_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.segmented_contrast_slider.setMinimum(-100)
        self.segmented_contrast_slider.setMaximum(100)
        self.segmented_contrast_slider.setValue(0)
        self.segmented_contrast_slider.valueChanged.connect(self.update_segmented_image)
        self.segmented_image_layout.addWidget(self.segmented_contrast_slider)

        self.image_layout.addLayout(self.segmented_image_layout)

    def init_depth_image_layout(self):
        self.depth_image_layout = QVBoxLayout()

        self.depth_export_layout = QHBoxLayout()
        self.depth_export_path = QLineEdit(self)
        self.depth_export_path.setPlaceholderText("Depth Image Export Path")
        self.depth_export_layout.addWidget(self.depth_export_path)
        self.depth_export_checkbox = QCheckBox("Export")
        self.depth_export_checkbox.setChecked(True)
        self.depth_export_layout.addWidget(self.depth_export_checkbox)
        self.depth_image_layout.addLayout(self.depth_export_layout)

        self.depth_image_label = QLabel('No Depth Image')
        self.depth_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_image_layout.addWidget(self.depth_image_label)
        self.depth_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_exposure_slider.setMinimum(-100)
        self.depth_exposure_slider.setMaximum(100)
        self.depth_exposure_slider.setValue(0)
        self.depth_exposure_slider.valueChanged.connect(self.update_depth_image)
        self.depth_image_layout.addWidget(self.depth_exposure_slider)

        self.depth_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_contrast_slider.setMinimum(-100)
        self.depth_contrast_slider.setMaximum(100)
        self.depth_contrast_slider.setValue(0)
        self.depth_contrast_slider.valueChanged.connect(self.update_depth_image)
        self.depth_image_layout.addWidget(self.depth_contrast_slider)

        self.image_layout.addLayout(self.depth_image_layout)

    def init_multiplied_image_layout(self):
        self.multiplied_image_layout = QVBoxLayout()

        self.multiplied_export_layout = QHBoxLayout()
        self.multiplied_export_path = QLineEdit(self)
        self.multiplied_export_path.setPlaceholderText("Multiplied Image Export Path")
        self.multiplied_export_layout.addWidget(self.multiplied_export_path)
        self.multiplied_export_checkbox = QCheckBox("Export")
        self.multiplied_export_checkbox.setChecked(True)
        self.multiplied_export_layout.addWidget(self.multiplied_export_checkbox)
        self.multiplied_image_layout.addLayout(self.multiplied_export_layout)

        self.multiplied_image_label = QLabel('No Multiplied Image')
        self.multiplied_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.multiplied_image_layout.addWidget(self.multiplied_image_label)
        self.multiplied_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.multiplied_exposure_slider.setMinimum(-100)
        self.multiplied_exposure_slider.setMaximum(100)
        self.multiplied_exposure_slider.setValue(0)
        self.multiplied_exposure_slider.valueChanged.connect(self.update_multiplied_image)
        self.multiplied_image_layout.addWidget(self.multiplied_exposure_slider)

        self.multiplied_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.multiplied_contrast_slider.setMinimum(-100)
        self.multiplied_contrast_slider.setMaximum(100)
        self.multiplied_contrast_slider.setValue(0)
        self.multiplied_contrast_slider.valueChanged.connect(self.update_multiplied_image)
        self.multiplied_image_layout.addWidget(self.multiplied_contrast_slider)

        self.image_layout.addLayout(self.multiplied_image_layout)

    def init_no_bg_image_layout(self):
        self.no_bg_image_layout = QVBoxLayout()

        self.no_bg_export_layout = QHBoxLayout()
        self.no_bg_export_path = QLineEdit(self)
        self.no_bg_export_path.setPlaceholderText("No Background Image Export Path")
        self.no_bg_export_layout.addWidget(self.no_bg_export_path)
        self.no_bg_export_checkbox = QCheckBox("Export")
        self.no_bg_export_checkbox.setChecked(True)
        self.no_bg_export_layout.addWidget(self.no_bg_export_checkbox)
        self.no_bg_image_layout.addLayout(self.no_bg_export_layout)

        self.no_bg_image_label = QLabel('No Background Image')
        self.no_bg_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_bg_image_layout.addWidget(self.no_bg_image_label)
        self.no_bg_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.no_bg_exposure_slider.setMinimum(-100)
        self.no_bg_exposure_slider.setMaximum(100)
        self.no_bg_exposure_slider.setValue(0)
        self.no_bg_exposure_slider.valueChanged.connect(self.update_no_bg_image)
        self.no_bg_image_layout.addWidget(self.no_bg_exposure_slider)

        self.no_bg_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.no_bg_contrast_slider.setMinimum(-100)
        self.no_bg_contrast_slider.setMaximum(100)
        self.no_bg_contrast_slider.setValue(0)
        self.no_bg_contrast_slider.valueChanged.connect(self.update_no_bg_image)
        self.no_bg_image_layout.addWidget(self.no_bg_contrast_slider)

        self.image_layout.addLayout(self.no_bg_image_layout)

    def init_sam_image_layout(self):
        self.sam_image_layout = QVBoxLayout()

        self.sam_export_layout = QHBoxLayout()
        self.sam_export_path = QLineEdit(self)
        self.sam_export_path.setPlaceholderText("SAM Image Export Path")
        self.sam_export_layout.addWidget(self.sam_export_path)
        self.sam_export_checkbox = QCheckBox("Export")
        self.sam_export_checkbox.setChecked(True)
        self.sam_export_layout.addWidget(self.sam_export_checkbox)
        self.sam_image_layout.addLayout(self.sam_export_layout)

        self.sam_image_label = QLabel('No SAM Image')
        self.sam_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sam_image_layout.addWidget(self.sam_image_label)
        self.sam_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.sam_exposure_slider.setMinimum(-100)
        self.sam_exposure_slider.setMaximum(100)
        self.sam_exposure_slider.setValue(0)
        self.sam_exposure_slider.valueChanged.connect(self.update_sam_image)
        self.sam_image_layout.addWidget(self.sam_exposure_slider)

        self.sam_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.sam_contrast_slider.setMinimum(-100)
        self.sam_contrast_slider.setMaximum(100)
        self.sam_contrast_slider.setValue(0)
        self.sam_contrast_slider.valueChanged.connect(self.update_sam_image)
        self.sam_image_layout.addWidget(self.sam_contrast_slider)

        self.image_layout.addLayout(self.sam_image_layout)

    def init_false_color_image_layout(self):
        self.false_color_image_layout = QVBoxLayout()

        self.false_color_export_layout = QHBoxLayout()
        self.false_color_export_path = QLineEdit(self)
        self.false_color_export_path.setPlaceholderText("False Color Image Export Path")
        self.false_color_export_layout.addWidget(self.false_color_export_path)
        self.false_color_export_checkbox = QCheckBox("Export")
        self.false_color_export_checkbox.setChecked(True)
        self.false_color_export_layout.addWidget(self.false_color_export_checkbox)
        self.false_color_image_layout.addLayout(self.false_color_export_layout)

        self.false_color_image_label = QLabel('No False Color Image')
        self.false_color_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.false_color_image_layout.addWidget(self.false_color_image_label)
        self.false_color_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.false_color_exposure_slider.setMinimum(-100)
        self.false_color_exposure_slider.setMaximum(100)
        self.false_color_exposure_slider.setValue(0)
        self.false_color_exposure_slider.valueChanged.connect(self.update_false_color_image)
        self.false_color_image_layout.addWidget(self.false_color_exposure_slider)

        self.false_color_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.false_color_contrast_slider.setMinimum(-100)
        self.false_color_contrast_slider.setMaximum(100)
        self.false_color_contrast_slider.setValue(0)
        self.false_color_contrast_slider.valueChanged.connect(self.update_false_color_image)
        self.false_color_image_layout.addWidget(self.false_color_contrast_slider)

        self.image_layout.addLayout(self.false_color_image_layout)

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Image Directory')
        if directory:
            self.image_files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                    try:
                        img = Image.open(file_path)
                        img.verify()
                        self.image_files.append(file_path)
                    except (IOError, SyntaxError) as e:
                        print(f"Skipping file {file}: {e}")
            if self.image_files:
                self.slider.setMaximum(len(self.image_files) - 1)
                self.current_index = 0
                self.slider.setValue(0)
                self.display_image()
            else:
                self.original_image_label.setText('No valid images found in the directory')
                self.segmented_image_label.setText('')
                self.depth_image_label.setText('')
                self.multiplied_image_label.setText('')
                self.no_bg_image_label.setText('')
                self.sam_image_label.setText('')
                self.false_color_image_label.setText('')

    def display_image(self):
        if self.image_files:
            try:
                image_path = self.image_files[self.current_index]
                image = Image.open(image_path)

                self.original_image = image.copy()
                self.adjusted_image = self.apply_exposure_contrast(image.copy(), self.original_exposure_slider.value(), self.original_contrast_slider.value())
                self.segmented_image, self.segmented_mask = self.segment_image(self.adjusted_image)
                self.depth_image = self.estimate_depth(self.adjusted_image)
                self.no_bg_image = self.remove_background(self.adjusted_image)
                self.sam_image_result = self.sam_image(self.adjusted_image)

                self.original_image_with_mask = self.apply_mask_to_image(self.original_image.copy(), self.segmented_mask)
                self.multiplied_image = self.multiply_images(self.segmented_image, self.depth_image)
                self.false_color_multiplied_image = self.apply_false_color(self.multiplied_image, self.original_image, self.segmented_mask)

                self.update_images()
            except Exception as e:
                print(f"Error displaying image: {e}")
                traceback.print_exc()
        else:
            self.original_image_label.setText('No Image Loaded')
            self.segmented_image_label.setText('')
            self.depth_image_label.setText('')
            self.multiplied_image_label.setText('')
            self.no_bg_image_label.setText('')
            self.sam_image_label.setText('')
            self.false_color_image_label.setText('')

    def update_images(self):
        if not self.image_files:
            return

        self.update_original_image()
        self.update_segmented_image()
        self.update_depth_image()
        self.update_multiplied_image()
        self.update_no_bg_image()
        self.update_sam_image()
        self.update_false_color_image()

    def display_qimage(self, image, label):
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pixmap)

    def slider_value_changed(self):
        self.current_index = self.slider.value()
        self.display_image()

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.setValue(self.current_index)
            self.display_image()

    def go_forward(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.slider.setValue(self.current_index)
            self.display_image()

    def resizeEvent(self, event):
        self.update_images()
        super().resizeEvent(event)

    def segment_image(self, image):
        try:
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame_resized = cv2.resize(frame, (512, 512))
            image_resized = Image.fromarray(frame_resized)

            inputs = self.seg_processor(images=image_resized, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
                logits = outputs.logits.cpu()

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
            color_palette = [
                (0, 0, 0),
                (128, 0, 0),
                (0, 128, 0),
                (128, 128, 0),
                (0, 0, 128),
                (128, 0, 128),
                (0, 128, 128),
                (128, 128, 128)
            ]

            color_palette = [
                (0, 0, 0),
                (128, 0, 0),
                (0, 0, 128),
                (0, 128, 0),
                (0, 128, 0),
                (0, 128, 0),
                (0, 128, 0),
                (0, 128, 0)
            ]
            color_segmented_image = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
            binary_mask = np.zeros((pred_seg.shape[0], pred_seg.shape[1]), dtype=np.uint8)

            for class_id in np.unique(pred_seg):
                if class_id == 0:
                    binary_mask[pred_seg == class_id] = 0  # Background
                else:
                    binary_mask[pred_seg == class_id] = 1  # Foreground
                    color_segmented_image[pred_seg == class_id] = color_palette[class_id % len(color_palette)]

            return Image.fromarray(color_segmented_image), binary_mask

        except Exception as e:
            print(f"An error occurred during image segmentation: {e}")
            traceback.print_exc()
            return Image.new("RGB", image.size), np.zeros(image.size[::-1], dtype=np.uint8)

    def estimate_depth(self, image):
        try:
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            image_resized = Image.fromarray(frame)

            inputs = self.depth_processor(images=image_resized, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_image = Image.fromarray(formatted)

            return depth_image

        except Exception as e:
            print(f"An error occurred during depth estimation: {e}")
            traceback.print_exc()
            return Image.new("L", image.size)

    def remove_background(self, image):
        try:
            model_input_size = [512, 512]  # Model input size can be adjusted as needed
            orig_im = np.array(image)
            orig_im_size = orig_im.shape[0:2]
            image_tensor = self.preprocess_image(orig_im, model_input_size).to(self.device)

            with torch.no_grad():
                result = self.no_bg_model(image_tensor)

            result_image = self.postprocess_image(result[0][0], orig_im_size)

            pil_im = Image.fromarray(result_image)
            no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
            no_bg_image.paste(image, mask=pil_im)
            return no_bg_image

        except Exception as e:
            print(f"An error occurred during background removal: {e}")
            traceback.print_exc()
            return image

    def sam_image(self, image):
        try:
            # Convert image to RGB format
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            raw_image = Image.fromarray(frame)

            # Define input points for localization
            input_points = [[[450, 600]]]

            # Process the image and get the mask
            inputs = self.sam_processor(raw_image, input_points=input_points, return_tensors="pt").to(self.device1)
            outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                                          inputs["original_sizes"].cpu(),
                                                                          inputs["reshaped_input_sizes"].cpu())

            # Assuming we are working with the first mask in the batch
            mask = masks[0][0].numpy()  # Get the first mask from the batch and convert to numpy

            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask[0]  # If mask has more than 2 dimensions, take the first one

            # Define color and apply it to the mask
            #color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]

            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # Convert the mask image to PIL Image
            mask_image = (mask_image * 255).astype(np.uint8)  # Scale values to [0, 255]
            mask_pil_image = Image.fromarray(mask_image)

            return mask_pil_image
        except Exception as e:
            print(f"An error occurred during SAM image processing: {e}")
            traceback.print_exc()
            return Image.new("RGBA", image.size)

    def preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    def apply_false_color(self, image, no_bg, mask):
        try:
            # Convert image to grayscale
            image_array = np.array(image.convert('L'))

            # Apply false color using the JET colormap
            false_color_image = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)

            # Ensure no_bg is a numpy array
            no_bg_array = np.array(no_bg)

            # Check if no_bg_array has the same shape as false_color_image
            if no_bg_array.shape != false_color_image.shape:
                raise ValueError("no_bg image must have the same dimensions as the input image.")

            # Ensure the mask is resized to match the false_color_image size
            mask = cv2.resize(mask, (false_color_image.shape[1], false_color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Set background pixels to black in the false color image
            false_color_image[mask == 0] = [0, 0, 0]

            # Convert the result back to a PIL Image and return
            return Image.fromarray(false_color_image)
        except Exception as e:
            print(f"An error occurred while applying false color: {e}")
            traceback.print_exc()
            return image

    def apply_mask_to_image(self, image, mask):
        try:
            frame = np.array(image.convert("RGBA"))
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            frame[mask == 0] = [0, 0, 0, 255]  # Set background pixels to black
            return Image.fromarray(frame)
        except Exception as e:
            print(f"An error occurred while applying the mask: {e}")
            traceback.print_exc()
            return image

    def multiply_images(self, seg_image, depth_image):
        try:
            seg_array = np.array(seg_image, dtype=np.float32) / 255
            depth_array = np.array(depth_image.convert("L"), dtype=np.float32) / 255

            depth_array_colored = np.stack([depth_array]*3, axis=-1)
            multiplied_array = cv2.multiply(seg_array, depth_array_colored)

            multiplied_array = cv2.normalize(multiplied_array, None, 0, 255, cv2.NORM_MINMAX)
            multiplied_image = Image.fromarray(multiplied_array.astype(np.uint8))
            return multiplied_image
        except Exception as e:
            print(f"An error occurred while multiplying images: {e}")
            traceback.print_exc()
            return Image.new("RGB", seg_image.size)

    def update_original_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.original_image_with_mask, self.original_exposure_slider.value(),
                                                      self.original_contrast_slider.value())
        self.display_qimage(enhanced_image, self.original_image_label)

    def update_segmented_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.segmented_image, self.segmented_exposure_slider.value(),
                                                      self.segmented_contrast_slider.value())
        self.display_qimage(enhanced_image, self.segmented_image_label)

    def update_depth_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.depth_image, self.depth_exposure_slider.value(),
                                                      self.depth_contrast_slider.value())
        self.display_qimage(enhanced_image, self.depth_image_label)

    def update_multiplied_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.multiplied_image, self.multiplied_exposure_slider.value(),
                                                      self.multiplied_contrast_slider.value())
        self.display_qimage(enhanced_image, self.multiplied_image_label)

    def update_no_bg_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.no_bg_image, self.no_bg_exposure_slider.value(),
                                                      self.no_bg_contrast_slider.value())
        self.display_qimage(enhanced_image, self.no_bg_image_label)

    def update_sam_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.sam_image_result, self.sam_exposure_slider.value(),
                                                      self.sam_contrast_slider.value())
        self.display_qimage(enhanced_image, self.sam_image_label)

    def update_false_color_image(self):
        if not self.image_files:
            return
        enhanced_image = self.apply_exposure_contrast(self.false_color_multiplied_image, self.false_color_exposure_slider.value(),
                                                      self.false_color_contrast_slider.value())
        self.display_qimage(enhanced_image, self.false_color_image_label)

    def apply_exposure_contrast(self, image, exposure_value, contrast_value):
        enhancer = ImageEnhance.Brightness(image)
        factor = (exposure_value + 100) / 100
        image = enhancer.enhance(factor)
        enhancer = ImageEnhance.Contrast(image)
        factor = (contrast_value + 100) / 100
        image = enhancer.enhance(factor)
        return image

    def export_images(self):
        if not self.image_files:
            return

        original_path = self.original_export_path.text()
        segmented_path = self.segmented_export_path.text()
        depth_path = self.depth_export_path.text()
        multiplied_path = self.multiplied_export_path.text()
        no_bg_path = self.no_bg_export_path.text()
        sam_path = self.sam_export_path.text()
        false_color_path = self.false_color_export_path.text()

        for index, image_path in enumerate(self.image_files):
            try:
                image = Image.open(image_path)
                original_image = image.copy()
                adjusted_image = self.apply_exposure_contrast(image.copy(), self.original_exposure_slider.value(),
                                                              self.original_contrast_slider.value())
                segmented_image, segmented_mask = self.segment_image(adjusted_image)
                depth_image = self.estimate_depth(adjusted_image)
                no_bg_image = self.remove_background(adjusted_image)
                sam_image_result = self.sam_image(adjusted_image)

                original_image_with_mask = self.apply_mask_to_image(original_image.copy(), segmented_mask)
                multiplied_image = self.multiply_images(segmented_image, depth_image)
                false_color_multiplied_image = self.apply_false_color(multiplied_image, original_image, segmented_mask)

                if self.original_export_checkbox.isChecked() and original_path:
                    original_export_file = os.path.join(original_path, f'original_{index}.png')
                    original_image_with_mask.save(original_export_file)

                if self.segmented_export_checkbox.isChecked() and segmented_path:
                    segmented_export_file = os.path.join(segmented_path, f'segmented_{index}.png')
                    segmented_image.save(segmented_export_file)

                if self.depth_export_checkbox.isChecked() and depth_path:
                    depth_export_file = os.path.join(depth_path, f'depth_{index}.png')
                    depth_image.save(depth_export_file)

                if self.multiplied_export_checkbox.isChecked() and multiplied_path:
                    multiplied_export_file = os.path.join(multiplied_path, f'multiplied_{index}.png')
                    multiplied_image.save(multiplied_export_file)

                if self.no_bg_export_checkbox.isChecked() and no_bg_path:
                    no_bg_export_file = os.path.join(no_bg_path, f'no_bg_{index}.png')
                    no_bg_image.save(no_bg_export_file)

                if self.sam_export_checkbox.isChecked() and sam_path:
                    sam_export_file = os.path.join(sam_path, f'sam_{index}.png')
                    sam_image_result.save(sam_export_file)

                if self.false_color_export_checkbox.isChecked() and false_color_path:
                    false_color_export_file = os.path.join(false_color_path, f'false_color_{index}.png')
                    false_color_multiplied_image.save(false_color_export_file)

            except Exception as e:
                print(f"An error occurred while exporting images: {e}")
                traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSlider()
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    window.show()
    sys.exit(app.exec())
