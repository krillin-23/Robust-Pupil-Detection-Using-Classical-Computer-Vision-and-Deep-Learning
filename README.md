# Robust Pupil Detection Using Classical Computer Vision and Deep Learning

## Part II — Transition to YOLOv8-Based Detection

A computer-vision and deep-learning pipeline for robust pupil detection in noisy eye images and videos, with downstream pupil-area analysis and Raspberry Pi deployment.

## Overview

This project is the sixth-semester continuation of a pupil-detection system originally developed using classical image processing and a U-Net model.

The earlier system combined Canny edge detection, contour analysis, and ellipse fitting with a U-Net that reconstructed ellipse-fitted pupil images for frames where the classical detector failed.

In this phase, the U-Net approach was replaced by YOLOv8n, a lightweight single-stage object detector that directly predicts a bounding box around the pupil. This removes the need for pixel-level reconstruction and provides a more compact output for downstream pupil-area estimation.

The project also extends the offline detection pipeline toward real-time Raspberry Pi deployment using the NCNN format of the trained YOLOv8n model.

The overall system estimates pupil area frame by frame and uses the temporal variation of pupil area for downstream fatigue analysis.

## Main Objectives

- Robustly detect the pupil in noisy eye images and videos.
- Handle challenging frames containing eyelashes, corneal reflections, and nearby dark structures.
- Estimate pupil area from the detected region.
- Process pre-recorded image folders and videos.
- Analyze pupil-area variation over time.
- Deploy the trained model on a Raspberry Pi using NCNN.
- Provide a GUI-based live detection system for Raspberry Pi.

## Methodology

The final pipeline consists of the following stages:

```text
Eye Images / Video
        │
        ▼
Classical Pupil Detection
        │
        ├── Successful detections
        │       │
        │       ▼
        │   YOLO training labels
        │
        ▼
YOLOv8n Training
        │
        ▼
best.pt
        │
        ├───────────────► PC / Offline Inference
        │
        ▼
NCNN Export
        │
        ▼
best_ncnn_model/
        │
        ▼
Raspberry Pi Deployment
        │
        ▼
Live Pupil Detection
        │
        ▼
Pupil Area vs. Time
        │
        ▼
Fatigue / Pupil Dynamics Analysis
```

## YOLOv8n Model

The project uses YOLOv8n, the nano variant of YOLOv8.

The detector is trained for a single class:

```text
pupil
```

The predicted bounding box is used to estimate pupil area by treating the detected box as an enclosing ellipse:

```text
A_pupil = (π / 4) × width × height
```

An axis-ratio filter is also applied:

```text
0.60 ≤ minor_axis / major_axis ≤ 1.00
```

This is used to reject geometrically implausible detections.

## Automatic Dataset Generation

The YOLO training data are generated automatically from the classical pupil-detection pipeline rather than being manually annotated.

The classical pipeline performs:

- Grayscale conversion
- Median filtering
- Morphological opening
- Canny edge detection
- Contour filtering
- Circularity testing
- Ellipse fitting

Successful pupil detections are converted into YOLO-format labels.

## Training

The training script is located at:

```text
training/1.py
```

The dataset configuration is:

```text
training/data_1.yaml
```

The label-generation script is:

```text
training/code_15.py
```

The trained checkpoint included in this repository is:

```text
models/best.pt
```

The project uses transfer learning from COCO-pretrained YOLOv8n weights.

## Model Performance

The final validation metrics reported in the project report are:

| Metric        | Value  |
|---------------|--------|
| Precision     | 0.9999 |
| Recall        | 1.0000 |
| mAP@0.5       | 0.9950 |
| mAP@0.5:0.95  | 0.9949 |
| F1 score      | 1.0000 |

The reported final training and validation losses are:

| Loss                | Train  | Validation |
|---------------------|--------|------------|
| Box loss            | 0.1865 | 0.1469     |
| Classification loss | 0.1533 | 0.1340     |
| DFL loss            | 0.7552 | 0.7471     |

## Offline Inference

The main PC inference pipeline is:

```text
inference/code_17.py
```

It performs YOLO inference on image folders, applies geometric filtering, draws an ellipse corresponding to the detected pupil, estimates pupil area, and performs downstream area and fatigue analysis.

## Raspberry Pi Deployment

The trained PyTorch model can be exported for Raspberry Pi deployment using NCNN.

The conversion utility is:

```text
deployment/convert_pt_to_ncnn.py
```

The exported model is stored in:

```text
models/best_ncnn_model/
```

The Raspberry Pi inference pipelines are:

```text
raspberry_pi/code_17_ncnn_v3.py
raspberry_pi/code_18_ncnn.py
```

`code_17_ncnn_v3.py` supports image-folder and video inputs.

`code_18_ncnn.py` provides a more production-oriented inference and analysis implementation.

## Live Raspberry Pi GUI

The GUI source is:

```text
gui/gui2.py
```

The documented live system uses:

- PiCamera2
- YOLOv8n with NCNN
- CustomTkinter
- Real-time pupil detection
- Pupil-area recording
- Fatigue analysis
- Graphical display of the live camera feed and analysis results

### Important Dependency Note

The supplied `gui2.py` imports the following helper modules:

```text
process_code.py
led_operation_10sec.py
```

These helper files were not present in the supplied project files and are therefore not included in this repository.

As a result, `gui2.py` is preserved as the supplied GUI source/reference implementation and should not be considered independently runnable without those missing helper modules.

## Pupil Area and Fatigue Analysis

The processing pipeline can generate frame-level and analysis outputs including:

```text
area.txt
zero_area_frames.csv
area_detected_frames.csv
recorded.mp4
slope_values.txt
area_plot.png
log_ellipse_fitting.log
```

The pupil-area signal is smoothed and its temporal slope is analysed as part of the fatigue-detection methodology.

## Project Structure

```text
Robust-Pupil-Detection-Using-Classical-Computer-Vision-and-Deep-Learning/
│
├── deployment/
│   └── convert_pt_to_ncnn.py
│
├── gui/
│   └── gui2.py
│
├── inference/
│   └── code_17.py
│
├── models/
│   ├── best.pt
│   └── best_ncnn_model/
│       ├── metadata.yaml
│       ├── model.ncnn.bin
│       └── model.ncnn.param
│
├── raspberry_pi/
│   ├── code_17_ncnn_v3.py
│   └── code_18_ncnn.py
│
├── report/
│   └── Project_Report.pdf
│
├── training/
│   ├── 1.py
│   ├── code_15.py
│   └── data_1.yaml
│
├── .gitignore
├── README.md
├── requirements-pi.txt
└── requirements.txt
```

## Installation

For the core Python scripts:

```text
pip install -r requirements.txt
```

For Raspberry Pi-specific dependencies:

```text
requirements-pi.txt
```

The Raspberry Pi environment additionally requires platform-specific camera and GPIO components.

## Running the Training Pipeline

Before training, update the dataset paths in:

```text
training/data_1.yaml
```

The supplied YAML file contains machine-specific paths and therefore must be adapted to the local dataset location before reproduction.

Then run:

```text
python training/1.py
```

## Running Offline Inference

The PC inference source is:

```text
inference/code_17.py
```

Before running it, update its model and input/output paths to match the local machine.

## Exporting the Model

The model-conversion utility is:

```text
deployment/convert_pt_to_ncnn.py
```

It exports the trained PyTorch checkpoint to a deployment format such as NCNN.

## Reproducibility Note

The repository preserves the supplied project source files and trained model.

Several scripts contain paths originating from the original development environment. These paths should be updated before running the scripts on another computer.

The supplied training script and the final project report also document different image-size configurations. The supplied training script currently specifies `imgsz=320`, whereas the report describes the final project configuration as 640 × 640 rectangular training. This should be checked before attempting to reproduce the exact reported training run.

## Dataset

The original pupil-image dataset is not included in this repository.

The YOLO training data were generated automatically from successful detections produced by the classical pupil-detection pipeline.

## Limitations

The project report identifies several limitations:

- The pupil is represented using an axis-aligned bounding box, which can introduce area-estimation error for tilted pupils.
- Detection performance may depend on the confidence threshold and geometric filtering threshold.
- The training data are derived from a limited set of subjects and a specific recording setup.
- Automatically generated labels inherit inaccuracies that may be present in the classical ellipse-fitting pipeline.
- The fatigue-analysis threshold requires further calibration across subjects and recording conditions.
- The live and offline pipelines use different fatigue slope thresholds and therefore require further validation for direct comparison.

## Future Work

The project report proposes several directions for further development:

- Oriented Bounding Box detection for improved geometric fitting.
- Instance segmentation for direct pixel-level pupil-area estimation.
- A unified and experimentally validated fatigue threshold.
- Training across more subjects and recording conditions.
- A learned end-to-end fatigue classifier based on pupil-dynamics features.

## Project Report

The complete project report is available here:

[Project Report](report/Project_Report.pdf)

## Author

**G Madhukar**

Department of Physics  
Indian Institute of Technology Hyderabad

## Supervisor

**Dr. Vandana Sharma**

Department of Physics  
Indian Institute of Technology Hyderabad

## Project Title

**Robust Pupil Detection Using Classical Computer Vision and Deep Learning: Part II – Transition to YOLOv8-Based Detection**

## License

See the repository license file for the applicable licensing information.
