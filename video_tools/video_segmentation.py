import cv2
import gc
import numpy as np
import pygetwindow as gw
import mss
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
import torch
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Call garbage collector
gc.collect()

# Empty the CUDA cache if using GPU
if device.type == 'cuda':
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

    # Initialize MaskFormer model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco").to(device)

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
            frame_resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert frame to PIL image
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Prepare inputs for the model
            inputs = feature_extractor(images=image, return_tensors="pt").to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process the outputs
            result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_panoptic_map = result["segmentation"]

            # Move the segmentation map to CPU
            predicted_panoptic_map = predicted_panoptic_map.cpu().numpy()

            # Convert segmentation map to color image for visualization
            color_segmented_image = np.zeros((predicted_panoptic_map.shape[0], predicted_panoptic_map.shape[1], 3), dtype=np.uint8)
            for class_id in np.unique(predicted_panoptic_map):
                color_segmented_image[predicted_panoptic_map == class_id] = class_id_to_color(class_id)

            # Display the frame and segmentation
            cv2.imshow('Window Capture', frame_resized)
            cv2.imshow('Instance Segmentation', color_segmented_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Replace 'Untitled - Notepad' with the title of the window you want to capture
window_title = '2024-06-01 17-24-24.mkv - VLC media player'
capture_window(window_title)
