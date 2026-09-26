# YOLOv8 Pupil Detection

This repository contains the source files, model files, and final project report supplied for the YOLOv8 pupil-detection project. The dataset is not included.

## Contents

- `code_15.py` generates labels from classical pupil detections.
- `Training/1.py` is the supplied YOLO training script.
- `code_17.py` processes image folders using the PyTorch model.
- `code_17_ncnn.py`, `code_17_ncnn_v3.py`, and `code_18_ncnn.py` contain inference variants.
- `archive_variants/code_17_ncnn_pi_headless.py` is a second, distinct headless/CSV variant found in the archive; it is retained here for comparison.
- `gui2.py` is the Raspberry Pi GUI source.
- `convert_pt_to_ncnn.py` exports the PyTorch checkpoint.
- `best.pt` and `best_ncnn_model/` contain the supplied trained model.
- `report/Project_Report.pdf` is the supplied 35-page report.

## Before running

The supplied scripts are preserved as received. Several configuration values still point to machine-specific dataset, model, or output folders. Update those paths for your machine and provide the dataset before running training or inference.

The supplied `Training/1.py` trains with `imgsz=320`; the report describes training with `imgsz=640` (report section 5.1), and the NCNN model metadata records 640x640. Confirm which training configuration is correct before using the script to reproduce the checkpoint.

The supplied `data_1.yaml` contains machine-specific paths and uses nested `images`/`labels` keys. The report describes an Ultralytics dataset layout with `images/train`, `images/val`, `labels/train`, and `labels/val`. Review and correct the YAML to match the actual dataset before training.

`gui2.py` imports `led_operation_10sec.py` and `process_code.py`, which were not present in the supplied archive. The GUI will need those helper modules before it can run. Raspberry Pi setup also needs platform-specific camera and GPIO dependencies.

The NCNN export folder includes its model parameter and binary files, metadata, and the generated `model_ncnn.py` helper. The helper contains hard-coded paths from another computer and is not needed by the Ultralytics NCNN runtime. The Python bytecode cache was excluded.

## Dependencies

The scripts use Ultralytics, OpenCV, NumPy, SciPy, Matplotlib, and (for the classical pipeline) openpyxl. The Pi GUI additionally imports CustomTkinter, Pillow, Picamera2, gpiozero, and the missing project helper modules. Install `requirements.txt` for the core scripts. `requirements-pi.txt` lists the additional pip packages imported by the GUI. Install Picamera2 through the Raspberry Pi OS package manager. The NCNN backend is provided through Ultralytics.

## Files intentionally left out

The archive also contained older or debug GUIs, `4.py`, a cache directory, generated sample videos/images/logs, and `Document from Varun.zip`. Those were left out. `Document from Varun.zip` is an exact duplicate of `best.pt` and was not copied.

## Privacy and licensing

This public repository includes the supplied report, which contains the student's name and ID. No project-level license file was supplied, so none has been added. The NCNN model metadata reports Ultralytics AGPL-3.0; review applicable terms before redistributing the model.

