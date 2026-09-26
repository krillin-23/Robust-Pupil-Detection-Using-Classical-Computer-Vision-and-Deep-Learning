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
    "ncnn"  : "/home/pi/Documents/codes/best_ncnn_model",
    "tflite": "/home/pi/model/best.tflite",
}

MODEL_PATH = MODEL_PATHS[MODEL_FORMAT]
CONF       = 0.25

# Ellipse shape filter — ratio of minor axis to major axis
# 1.0 = perfect circle, lower = more oval
MIN_AXIS_RATIO = 0.60
MAX_AXIS_RATIO = 1.0

# ====== Paths ======
# INPUT_SOURCES: List of inputs — each entry can be:
#   - A path to a VIDEO FILE  (e.g. "/home/pi/videos/session1.mp4")
#   - A path to an IMAGE FOLDER (e.g. "/home/pi/frames/frames_6")
#
# Both types can be mixed freely in the same list.
INPUT_SOURCES = [
    "/home/pi/frames/frames_6"      # ← replace with your video or image folder path
]

BASE_OUTPUT = "/home/pi/Documents/YOLO/Outputs"


# ---------------------------------------------------------------
# INPUT TYPE DETECTOR
# Returns "video" if the path is a video file,
# "images" if the path is a folder of images,
# or raises an error if neither.
# ---------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def detect_input_type(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return "video"
        raise ValueError(
            f"Unsupported file type '{ext}' for: {input_path}\n"
            f"Supported video formats: {VIDEO_EXTENSIONS}"
        )
    elif os.path.isdir(input_path):
        return "images"
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


# ---------------------------------------------------------------
# FRAME ITERATOR
# Yields (frame_index, frame_bgr) for both videos and image folders.
# For videos, FRAME_RATE is read from the file metadata.
# ---------------------------------------------------------------
def frame_iterator(input_path, input_type):
    """
    Generator that yields (frame_idx, frame) tuples.
    Also returns the detected frame rate as a second value via a mutable dict.
    """
    if input_type == "video":
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"[WARNING] Could not read FPS from video, defaulting to {FRAME_RATE}")
            fps = FRAME_RATE

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame, fps
            idx += 1

        cap.release()

    else:  # images folder
        images = sorted([
            f for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if not images:
            raise FileNotFoundError(f"No images found in folder: {input_path}")

        for idx, img_file in enumerate(images):
            frame = cv2.imread(os.path.join(input_path, img_file))
            if frame is None:
                print(f"[WARNING] Could not read image: {img_file}, skipping.")
                continue
            yield idx, frame, FRAME_RATE   # image folders always use the configured FRAME_RATE


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


def create_workspace(input_name):
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    workspace_name = f"{input_name}_Ellipse_{timestamp}"
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
def process_single_source(input_path):
    # ── Detect whether input is a video file or an image folder ──────────────
    input_type = detect_input_type(input_path)

    # Derive a clean name for workspace/log naming
    input_name = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]

    print(f"--- Processing [{input_type.upper()}]: {input_name} ---")

    workspace = create_workspace(input_name)
    logger = setup_logger(workspace)

    # Log which model format and input type are being used
    logger.info(f"Input type : {input_type} | Path: {input_path}")
    logger.info(f"Model format: {MODEL_FORMAT} | Path: {MODEL_PATH}")

    area_txt_path = os.path.join(workspace, "area.txt")

    area_list, time_list = [], []
    prev_ellipse = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None   # initialised on first frame (so we know the real FPS)

    # ── CSV writers ───────────────────────────────────────────────────────────
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
    active_fps  = FRAME_RATE       # will be updated from the first frame's fps value

    with open(area_txt_path, 'w') as f:
        f.write("Area, Frame, Time\n")

    for frame_idx, frame, fps in frame_iterator(input_path, input_type):

        # Capture the real FPS on the very first frame
        if frame_idx == 0:
            active_fps = fps
            print(f"[INFO] Using frame rate: {active_fps:.2f} FPS")
            video_writer = cv2.VideoWriter(
                os.path.join(workspace, "recorded.mp4"),
                fourcc,
                active_fps,
                DISPLAY_RESOLUTION
            )

        frame   = cv2.resize(frame, DISPLAY_RESOLUTION)
        img_h, img_w = frame.shape[:2]
        elapsed = frame_idx / active_fps

        latest_area      = 0.0
        ellipse_detected = False

        # ── CHANGED: verbose is now set to True to show processing speeds ────
        results = model.predict(source=frame, conf=CONF, save=False, verbose=True)

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
                    logger.info(f"Skipped frame {frame_idx} — axis ratio {axis_ratio:.2f} out of range")
                    print(f"Skipped frame {frame_idx} — axis ratio {axis_ratio:.2f} out of range")
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

        if video_writer:
            video_writer.write(frame)

        if DISPLAY_AVAILABLE:
            cv2.imshow("Ellipse Detection with YOLO + Axis Ratio Filter", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                print("User interrupted processing.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    zero_csv_file.close()
    detected_csv_file.close()

    if video_writer:
        video_writer.release()
    if DISPLAY_AVAILABLE:
        cv2.destroyAllWindows()

    logger.info("Processing complete. Starting plot.")
    plot_area_vs_time(area_list, time_list, workspace)

    print(f"--- Finished processing {input_name} ---\n")


def main():
    for source in INPUT_SOURCES:
        process_single_source(source)
    print("All sources processed successfully!")


if __name__ == "__main__":
    main()