# Computer Vision Assignment

This repository contains Python demonstrations of fundamental image and video
processing techniques using OpenCV.

## Topics

| Script | Description | Input | Output directory |
| --- | --- | --- | --- |
| `Filtering-1.py` | Adds salt-and-pepper and Gaussian noise, applies mean, median, and Gaussian filters, and compares results. | `Turtle.jpg` | `filter_outputs/` |
| `Multi-Stage-Boundary-Extraction.py` | Detects edges with Canny and corners with the Harris detector. | `Turtle.jpg` | `edge_corner_outputs/` |
| `MathematicalMorphology.py` | Demonstrates binary morphology, silhouette and boundary extraction, and 8-directional chain codes. | `Turttle-Black-and-White.jpg` | `morphology_outputs/` |
| `Otsu-Watershed-Segmentation.py` | Segments coins using Otsu thresholding and watershed markers. | `Coins.jpg` | `segmentation_outputs_final/` |
| `YOLO_Nano_Video_Object_Detection.py` | Runs YOLO object detection on a video and saves an annotated video. | `Simple Street Scene.mp4`, `yolo11n.pt` | `object_detection_outputs/` |

The scripts expect their input files in the project directory. Generated output
directories are created automatically when the scripts run.

## Requirements

- Python
- The packages listed in [`requirements.txt`](requirements.txt)

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run a demonstration

From the project directory, run the script you want to explore:

```powershell
python Filtering-1.py
python Multi-Stage-Boundary-Extraction.py
python MathematicalMorphology.py
python Otsu-Watershed-Segmentation.py
python YOLO_Nano_Video_Object_Detection.py
```

The YOLO video script opens a display window; press **Q** or **Esc** to stop
processing. Its annotated video is saved under `object_detection_outputs/`.
