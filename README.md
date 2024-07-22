# AI Tools

This repository contains a collection of tools for processing images and videos, including generating depth maps and masks. The tools are organized into different categories such as `image_tools` and `video_tools`.

## Directory Structure



## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AI-tools.git
    cd AI-tools
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Image Tools

- **image_to_masks.py**: Converts an image to masks.

    Usage:
    ```bash
    python image_tools/image_to_masks.py --input <input_image_path> --output <output_mask_path>
    ```

### Video Tools

- **video_clothes_mask.py**: Generates masks for clothes in a video.

    Usage:
    ```bash
    python video_tools/video_clothes_mask.py --input <input_video_path> --output <output_path>
    ```

- **video_depth_mask.py**: Generates depth masks for a video.

    Usage:
    ```bash
    python video_tools/video_depth_mask.py --input <input_video_path> --output <output_path>
    ```

- **video_object.py**: Processes video to identify and mask objects.

    Usage:
    ```bash
    python video_tools/video_object.py --input <input_video_path> --output <output_path>
    ```

- **video_segmentation.py**: Segments different parts of the video.

    Usage:
    ```bash
    python video_tools/video_segmentation.py --input <input_video_path> --output <output_path>
    ```

- **video_segmentation_clothes.py**: Segments clothes in the video.

    Usage:
    ```bash
    python video_tools/video_segmentation_clothes.py --input <input_video_path> --output <output_path>
    ```

- **video_segmentation_everything.py**: Segments all parts of the video.

    Usage:
    ```bash
    python video_tools/video_segmentation_everything.py --input <input_video_path> --output <output_path>
    ```

- **video_to_images.py**: Converts video frames to images.

    Usage:
    ```bash
    python video_tools/video_to_images.py --input <input_video_path> --output <output_folder>
    ```

## Utilities

- **GPU.py**: Contains utilities for GPU processing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors and the open-source community for their invaluable work and support.
