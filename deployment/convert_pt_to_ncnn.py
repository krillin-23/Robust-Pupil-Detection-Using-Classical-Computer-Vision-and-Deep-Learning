"""
export_model.py
===============
Run this ONCE on your PC to convert your .pt model to NCNN or TFLite.
Then copy the exported folder/file to your Raspberry Pi.

Usage:
    python export_model.py
"""

from ultralytics import YOLO
import os

# ====== CONFIGURE THESE ======
PT_MODEL_PATH = r"C:\Users\madhu\Desktop\Pupil project\YOLO\best.pt"

# Choose export format: "ncnn" (recommended for Pi) or "tflite"
EXPORT_FORMAT = "ncnn"

# Image size used during training (keep same for export)
IMGSZ = 640
# ==============================


def export():
    if not os.path.isfile(PT_MODEL_PATH):
        print(f"ERROR: Model not found at {PT_MODEL_PATH}")
        return

    print(f"Loading model: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)

    print(f"Exporting to format: {EXPORT_FORMAT} ...")

    if EXPORT_FORMAT == "ncnn":
        # Exports to a folder named: best_ncnn_model/
        # Copy this entire folder to Raspberry Pi
        export_path = model.export(
            format="ncnn",
            imgsz=IMGSZ,
        )
        print(f"\n✅ NCNN export complete!")
        print(f"   Exported folder : {export_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Copy the folder '{export_path}' to your Raspberry Pi")
        print(f"   2. In code_17_ncnn.py, set:")
        print(f"      MODEL_FORMAT = 'ncnn'")
        print(f"      MODEL_PATHS['ncnn'] = '<path on Pi>/best_ncnn_model'")

    elif EXPORT_FORMAT == "tflite":
        # Exports to a file named: best.tflite
        # Copy this file to Raspberry Pi
        export_path = model.export(
            format="tflite",
            imgsz=IMGSZ,
            int8=False,   # Set True for further quantization (faster, slight accuracy loss)
        )
        print(f"\n✅ TFLite export complete!")
        print(f"   Exported file   : {export_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Copy '{export_path}' to your Raspberry Pi")
        print(f"   2. In code_17_ncnn.py, set:")
        print(f"      MODEL_FORMAT = 'tflite'")
        print(f"      MODEL_PATHS['tflite'] = '<path on Pi>/best.tflite'")

    else:
        print(f"Unknown EXPORT_FORMAT: {EXPORT_FORMAT}. Use 'ncnn' or 'tflite'.")


if __name__ == "__main__":
    export()
