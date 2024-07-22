import requests, gc
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Call garbage collector
gc.collect()

# Empty the CUDA cache if using GPU

torch.cuda.empty_cache()

# Load model and processor
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Load and resize image
img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
resized_image = raw_image.resize((raw_image.width // 2, raw_image.height // 2))

# Define input points
input_points = [[[450, 600]]]  # 2D localization of a window

# Set up the pipeline for mask generation
generator = pipeline(
    "mask-generation",
    model=model,
    image_processor=processor.image_processor,
    device=0
)

# Generate masks
outputs = generator(resized_image, points_per_batch=256)

# Function to display the mask
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Display the image and mask
plt.imshow(np.array(resized_image))
ax = plt.gca()
for mask in outputs['masks']:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
plt.show()
