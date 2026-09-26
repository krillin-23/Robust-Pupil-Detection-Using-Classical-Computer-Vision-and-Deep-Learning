import cv2
import numpy as np
import os
import csv
import logging
from datetime import datetime
from ultralytics import YOLO
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')  # Headless backend — no display required (required for Raspberry Pi)
import matplotlib.pyplot as plt

# ====== Display Mode ======
# Automatically detects whether a display is available (e.g. monitor on Pi).
# Set to False manually if you are running headless with no screen.
DISPLAY_AVAILABLE = os.environ.get("DISPLAY", "") != ""

# ====== Configuration ======
FRAME_RATE = 30.0
DISPLAY_RESOLUTION = (640, 480)
ROI_EXPAND = 80

# ====== YOLO Configuration ======
# ---------------------------------------------------------------
# MODEL FORMAT SELECTOR
# Set MODEL_FORMAT to one of:
#   "pt"    — original PyTorch model (use on PC)
#   "ncnn"  — NCNN format (FASTEST on Raspberry Pi, recommended)
#   "tflite"— TFLite format (alternative for Pi)
#
# HOW TO EXPORT (run export_model.py ONCE on your PC before
# copying the exported model folder to Raspberry Pi):
#   NCNN   → produces a folder:  best_ncnn_model/
#   TFLite → produces a file:    best.tflite
# ---------------------------------------------------------------
MODEL_FORMAT = "ncnn"   # <-- Change to "pt" / "ncnn" / "tflite"

# Paths for each format — update these to match your file locations on the Pi
MODEL_PATHS = {
    "pt"    : "/home/pi/model/best.pt",
    "ncnn"  : "/home/pi/model/best_ncnn_model",
    "tflite": "/home/pi/model/best.tflite",
}

MODEL_PATH = MODEL_PATHS[MODEL_FORMAT]
CONF       = 0.25

# Ellipse shape filter — ratio of minor axis to major axis
# 1.0 = perfect circle, lower = more oval
MIN_AXIS_RATIO = 0.60
MAX_AXIS_RATIO = 1.0

# ====== Paths ======
INPUT_FOLDERS = [
    "/home/pi/frames/frames_6"
]

BASE_OUTPUT = "/home/pi/outputs"


# ---------------------------------------------------------------
# MODEL LOADER
# Handles all three formats transparently.
# NCNN requires the folder path (e.g. best_ncnn_model/)
# TFLite requires the .tflite file path
# PT requires the .pt file path
# ---------------------------------------------------------------
def load_model(model_path, model_format):
    print(f"[MODEL] Loading format='{model_format}' from: {model_path}")

    if model_format == "ncnn":
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"NCNN model folder not found: {model_path}\n"
                f"Run export_model.py on your PC first to generate it."
            )
        model = YOLO(model_path)

    elif model_format == "tflite":
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model file not found: {model_path}\n"
                f"Run export_model.py on your PC first to generate it."
            )
        model = YOLO(model_path)

    elif model_format == "pt":
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PT model not found: {model_path}")
        model = YOLO(model_path)

    else:
        raise ValueError(f"Unknown MODEL_FORMAT: '{model_format}'. Use 'pt', 'ncnn', or 'tflite'.")

    print(f"[MODEL] Loaded successfully.")
    return model


# Load YOLO model once at startup
model = load_model(MODEL_PATH, MODEL_FORMAT)


def create_workspace(input_folder_name):
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    workspace_name = f"{input_folder_name}_Ellipse_{timestamp}"
    workspace = os.path.join(BASE_OUTPUT, workspace_name)
    os.makedirs(workspace, exist_ok=True)
    return workspace


def setup_logger(folder_name):
    logger = logging.getLogger(f"EllipseLogger_{os.path.basename(folder_name)}")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.FileHandler(os.path.join(folder_name, "log_ellipse_fitting.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


# ===============================
# FATIGUE ANALYSIS + PLOT
# ===============================
def plot_area_vs_time(area_list, time_list, workspace):

    if len(area_list) < 20:
        return

    area = np.array(area_list)
    time_arr = np.array(time_list)

    # ===== Smoothing =====
    window_size = min(11, len(area) - 1)
    if window_size % 2 == 0:
        window_size -= 1

    if window_size >= 5:
        smoothed_area = savgol_filter(area, window_size, 3)
    else:
        smoothed_area = area

    # ===== Derivative =====
    slope_val = np.diff(smoothed_area) / np.diff(time_arr)

    # Save instantaneous slope values
    with open(os.path.join(workspace, "slope_values.txt"), "w") as f:
        for s in slope_val:
            f.write(f"{s}\n")

    # ===== Segment Detection =====
    segments = []
    current_segment = [0]

    for i in range(1, len(slope_val)):
        if (slope_val[i] > 0 and slope_val[i - 1] < 0) or \
           (slope_val[i] < 0 and slope_val[i - 1] > 0):

            if len(current_segment) >= 50:
                segments.append(current_segment)

            current_segment = [i]
        else:
            current_segment.append(i)

    if len(current_segment) >= 50:
        segments.append(current_segment)

    # ===== Plotting =====
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, smoothed_area, label="Smoothed Area", color="black")

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    most_negative_slope = 0
    best_segment = None

    for idx, segment in enumerate(segments):

        t_seg = [time_arr[i + 1] for i in segment]
        a_seg = [smoothed_area[i + 1] for i in segment]

        if len(t_seg) > 1:
            slope, intercept, r_value, _, _ = linregress(t_seg, a_seg)

            if slope < most_negative_slope:
                most_negative_slope = slope
                best_segment = segment

            plt.scatter(
                t_seg,
                a_seg,
                color=colors[idx % len(colors)],
                s=35,
                label=f"Segment {idx + 1}: slope={slope:.2f}"
            )

    if most_negative_slope < -500:
        fatigue_msg = "PERSON IS NOT FATIGUED"
        fatigue_color = "green"
    else:
        fatigue_msg = "PERSON IS FATIGUED"
        fatigue_color = "red"

    plt.title("Area vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Area")
    plt.grid(True)
    plt.legend()

    plt.text(
        0.05,
        0.95,
        f"{fatigue_msg}\n(Most Negative Regression Slope = {most_negative_slope:.2f})",
        transform=plt.gca().transAxes,
        fontsize=12,
        color=fatigue_color,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')
    )

    plt.savefig(os.path.join(workspace, "area_plot.png"))
    # plt.show() removed — no display on headless Raspberry Pi
    plt.close()


# ===============================
# MAIN PROCESSING FUNCTION
# ===============================
def process_single_folder(input_folder):
    folder_name = os.path.basename(os.path.normpath(input_folder))
    print(f"--- Processing folder: {folder_name} ---")

    workspace = create_workspace(folder_name)
    logger = setup_logger(workspace)

    # Log which model format is being used
    logger.info(f"Model format: {MODEL_FORMAT} | Path: {MODEL_PATH}")

    area_txt_path = os.path.join(workspace, "area.txt")

    if not os.path.exists(input_folder):
        print(f"Error: Folder {input_folder} does not exist.")
        return

    images = sorted([f for f in os.listdir(input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    if not images:
        print(f"No images found in {input_folder}.")
        return

    frame_idx = 0
    area_list, time_list = [], []
    prev_ellipse = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        os.path.join(workspace, "recorded.mp4"),
        fourcc,
        FRAME_RATE,
        DISPLAY_RESOLUTION
    )

    # ── CSV writers (no Excel, no image folders) ─────────────────────────────
    zero_csv_path     = os.path.join(workspace, "zero_area_frames.csv")
    detected_csv_path = os.path.join(workspace, "area_detected_frames.csv")

    zero_csv_file     = open(zero_csv_path,     'w', newline='')
    detected_csv_file = open(detected_csv_path, 'w', newline='')

    zero_writer     = csv.writer(zero_csv_file)
    detected_writer = csv.writer(detected_csv_file)

    zero_writer.writerow(["Serial No", "Frame No", "Time (s)", "Area"])
    detected_writer.writerow(["Serial No", "Frame No", "Time (s)", "Area"])
    # ─────────────────────────────────────────────────────────────────────────

    zero_serial = detected_serial = 1

    with open(area_txt_path, 'w') as f:
        f.write("Area, Frame, Time\n")

    for img_file in images:
        img_path = os.path.join(input_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, DISPLAY_RESOLUTION)
        img_h, img_w = frame.shape[:2]
        elapsed = frame_idx / FRAME_RATE

        latest_area      = 0.0
        ellipse_detected = False

        # ── YOLO Detection ───────────────────────────────────────────────────
        # Works identically for .pt, NCNN, and TFLite —
        # Ultralytics handles the format difference internally.
        results = model.predict(source=frame, conf=CONF, save=False, verbose=False)

        for result in results:
            for box in result.boxes:
                x_center_n, y_center_n, box_w_n, box_h_n = box.xywhn[0].tolist()

                x_center_px = x_center_n * img_w
                y_center_px = y_center_n * img_h
                box_w_px    = box_w_n    * img_w
                box_h_px    = box_h_n    * img_h

                # Axis ratio filter
                minor_axis = min(box_w_px, box_h_px)
                major_axis = max(box_w_px, box_h_px)
                axis_ratio = minor_axis / major_axis if major_axis > 0 else 0

                if not (MIN_AXIS_RATIO <= axis_ratio <= MAX_AXIS_RATIO):
                    logger.info(f"Skipped {img_file} — axis ratio {axis_ratio:.2f} out of range")
                    print(f"Skipped {img_file} — axis ratio {axis_ratio:.2f} out of range")
                    continue

                MA = box_w_px
                ma = box_h_px
                x  = x_center_px
                y  = y_center_px

                area = (np.pi / 4) * MA * ma

                center = (int(x), int(y))
                axes   = (int(MA / 2), int(ma / 2))
                cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)

                conf_score = float(box.conf[0])
                cv2.putText(
                    frame,
                    f"conf:{conf_score:.2f} ratio:{axis_ratio:.2f}",
                    (center[0] - 20, center[1] - axes[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                with open(area_txt_path, 'a') as f:
                    f.write(f"{area:.2f}, {frame_idx}, {elapsed:.2f}\n")

                area_list.append(area)
                time_list.append(elapsed)

                latest_area      = area
                prev_ellipse     = (x, y, MA)
                ellipse_detected = True
                break

            if ellipse_detected:
                break
        # ─────────────────────────────────────────────────────────────────────

        if not ellipse_detected:
            prev_ellipse = None

        if latest_area == 0.0:
            zero_writer.writerow([zero_serial, frame_idx, round(elapsed, 2), latest_area])
            zero_serial += 1
        else:
            detected_writer.writerow([detected_serial, frame_idx, round(elapsed, 2), round(latest_area, 2)])
            detected_serial += 1

        overlay_text = f"Frame: {frame_idx} | Time: {elapsed:.2f}s | Area: {latest_area:.1f}"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        video_writer.write(frame)
        if DISPLAY_AVAILABLE:
            cv2.imshow("Ellipse Detection with YOLO + Axis Ratio Filter", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                print("User interrupted processing.")
                break

        frame_idx += 1

    # Close CSV files
    zero_csv_file.close()
    detected_csv_file.close()

    video_writer.release()
    if DISPLAY_AVAILABLE:
        cv2.destroyAllWindows()

    logger.info("Processing complete. Starting plot.")
    plot_area_vs_time(area_list, time_list, workspace)

    print(f"--- Finished processing {folder_name} ---\n")


def main():
    for folder in INPUT_FOLDERS:
        process_single_folder(folder)
    print("All folders processed successfully!")


if __name__ == "__main__":
    main()
