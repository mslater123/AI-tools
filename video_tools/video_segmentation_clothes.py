import cv2
import gc
import numpy as np
import pygetwindow as gw
import mss
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch
import torch.nn as nn
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Call garbage collector
gc.collect()

# Empty the CUDA cache if using GPU
torch.cuda.empty_cache()

# Function to map class IDs to colors for visualization
def class_id_to_color(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 256, 3))

# Function to capture the window
def capture_window(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window with title '{window_title}' not found.")
        return

    window = windows[0]

    # Initialize Segformer model
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(device)

    with mss.mss() as sct:
        while True:
            # Define the region to capture
            region = {
                "top": window.top,
                "left": window.left,
                "width": window.width,
                "height": window.height
            }

            # Capture the region
            screenshot = sct.grab(region)

            # Convert the screenshot to a numpy array
            frame = np.array(screenshot)

            # Convert the color from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize frame to 25% of its original size
            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert frame to PIL image
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Prepare inputs for the model
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu()

            # Upsample the logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            # Get the predicted segmentation map
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

            # Convert segmentation map to color image for visualization
            color_segmented_image = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
            for class_id in np.unique(pred_seg):
                color_segmented_image[pred_seg == class_id] = class_id_to_color(class_id)

            # Display the frame and segmentation
            cv2.imshow('Window Capture', frame_resized)
            cv2.imshow('Semantic Segmentation', color_segmented_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Replace 'Untitled - Notepad' with the title of the window you want to capture
window_title = 'window.mp4 - VLC media player'
capture_window(window_title)
