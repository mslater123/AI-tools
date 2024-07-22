import cv2
import gc
import numpy as np
import pygetwindow as gw
import mss
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Call garbage collector
gc.collect()

# Empty the CUDA cache if using GPU
torch.cuda.empty_cache()


# Function to capture the window
def capture_window(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window with title '{window_title}' not found.")
        return

    window = windows[0]

    # Initialize depth estimation model
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

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

            # Estimate depth on resized frame
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # Convert depth prediction to image
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_image = Image.fromarray(formatted)

            # Display the frame and depth
            cv2.imshow('Window Capture', frame_resized)
            cv2.imshow('Depth Estimation', np.array(depth_image))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


# Replace 'Untitled - Notepad' with the title of the window you want to capture
#window_title = 'Alina_1080.mp4 - VLC media player'
window_title = '2024-06-01 17-24-24.mkv - VLC media player'
capture_window(window_title)
