import cv2
import gc
import numpy as np
import pygetwindow as gw
import mss
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Call garbage collector
gc.collect()

# Empty the CUDA cache if using GPU
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Function to capture the window
def capture_window(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window with title '{window_title}' not found.")
        return

    window = windows[0]

    # Initialize object detection model
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small-300")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small-300").to(device)

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

            # Object detection on resized frame
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)

            # Process the detections
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label = model.config.id2label[label.item()]
                score = round(score.item(), 3)

                # Draw bounding box
                cv2.rectangle(frame_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # Put label
                cv2.putText(frame_resized, f"{label}: {score}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Window Capture', frame_resized)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Replace 'Untitled - Notepad' with the title of the window you want to capture
window_title = 'window.mkv - VLC media player'
capture_window(window_title)
