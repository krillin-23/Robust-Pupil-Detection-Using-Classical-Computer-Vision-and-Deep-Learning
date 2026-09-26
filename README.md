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
