# Use the 10-second LED operation sequence instead of the default one.
from led_operation_10sec import start_leds, stop_leds, ir_on_idle, ir_off
import customtkinter as ctk
import threading
import json
import sys
import cv2
import time
import os
import numpy as np
from datetime import datetime
from PIL import Image
import tkinter.messagebox as mbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from picamera2 import Picamera2
from gpiozero import LED
from scipy.signal import savgol_filter
from scipy.stats import linregress
from process_code import (create_workspace, setup_logger)
from typing import Optional, Tuple

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as exc:
    raise RuntimeError(
        "ultralytics is required for YOLO pupil detection (no fallback enabled)"
    ) from exc

RECORD_DURATION = 12  # seconds
FRAME_RATE = 30.0
DISPLAY_RESOLUTION = (320, 240)

# ── YOLO pupil detection (NCNN model folder) ─────────────────────────────────
YOLO_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "best_ncnn_model")
YOLO_CONF: float = 0.25
YOLO_IMG_SIZE: int = 320
YOLO_MIN_AXIS_RATIO: float = 0.60
YOLO_MAX_AXIS_RATIO: float = 1.00
YOLO_CLASS_ID: int = 0  # "pupil" in best_ncnn_model/metadata.yaml


class YoloPupilDetector:
    def __init__(
        self,
        *,
        model_path: str = YOLO_MODEL_PATH,
        conf: float = YOLO_CONF,
        imgsz: int = YOLO_IMG_SIZE,
        class_id: int = YOLO_CLASS_ID,
        min_axis_ratio: float = YOLO_MIN_AXIS_RATIO,
        max_axis_ratio: float = YOLO_MAX_AXIS_RATIO,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.class_id = int(class_id)
        self.min_axis_ratio = float(min_axis_ratio)
        self.max_axis_ratio = float(max_axis_ratio)

    def detect_best(
        self,
        frame_bgr: np.ndarray,
        *,
        prev_center_xy: Optional[Tuple[float, float]] = None,
        prev_center_max_jump_px: float = 160.0,
    ) -> tuple[
        Optional[Tuple[float, float]],
        Optional[Tuple[float, float]],
        float,
        float,
    ]:
        """
        Returns
        -------
        center_xy : (x, y) in pixels, or None if no pupil found
        axes_xy   : (axis1, axis2) (half-lengths) in pixels, or None
        area_px   : ellipse area estimate
        conf_best  : confidence for the chosen detection
        """
        h, w = frame_bgr.shape[:2]
        best_area = 0.0
        best_conf = 0.0
        best_center: Optional[Tuple[float, float]] = None
        best_axes: Optional[Tuple[float, float]] = None
        best_score = -1.0

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            save=False,
            device="cpu",
        )

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item()) if box.cls is not None and len(box.cls) else -1
                if cls != self.class_id:
                    continue

                # Use xyxy pixel coords to avoid any ambiguity from letterboxing.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_w_px = float(x2 - x1)
                box_h_px = float(y2 - y1)
                if box_w_px <= 1.0 or box_h_px <= 1.0:
                    continue

                x_center_px = float((x1 + x2) * 0.5)
                y_center_px = float((y1 + y2) * 0.5)

                minor_axis = min(box_w_px, box_h_px)
                major_axis = max(box_w_px, box_h_px)
                axis_ratio = (minor_axis / major_axis) if major_axis > 0 else 0.0
                if not (self.min_axis_ratio <= axis_ratio <= self.max_axis_ratio):
                    continue

                conf_score = float(box.conf[0].item()) if box.conf is not None and len(box.conf) else 0.0

                # Ellipse area estimate from YOLO bbox axes (diameters).
                area = float((np.pi / 4.0) * box_w_px * box_h_px)

                # cv2.ellipse expects half-length axes.
                axes1 = max(1.0, box_w_px / 2.0)
                axes2 = max(1.0, box_h_px / 2.0)

                # Prefer the most "pupil-like" box:
                #  - largest ellipse area (bbox-driven)
                #  - high confidence
                #  - (optional) proximity to last center
                base_score = conf_score * area
                if prev_center_xy is None:
                    score = base_score
                else:
                    dx = x_center_px - float(prev_center_xy[0])
                    dy = y_center_px - float(prev_center_xy[1])
                    dist = float(np.hypot(dx, dy))
                    score = base_score / (1.0 + dist / max(1.0, prev_center_max_jump_px))

                if score > best_score:
                    best_score = score
                    best_conf = conf_score
                    best_area = area
                    best_center = (x_center_px, y_center_px)
                    best_axes = (axes1, axes2)

        return best_center, best_axes, best_area, best_conf


class AppState:
    def __init__(self):
        self.gui_alive = True
        self.masking_enabled = False
        self.stop_requested = False
        self.after_id = None
        self.video_active = True
        self.recording_active = False
        self.recording_start_time = None

state = AppState()

camera_instance = None

def reset_state():
    global state
    state = AppState()
    state.gui_alive = True

def safe_close(root):
    try:
        state.gui_alive = False
        if root.winfo_exists():
            root.after(0, root.destroy)
    except Exception as e:
        print("Safe close error:", e)


def back_to_main(root):
    state.stop_requested = True
    state.gui_alive = False
    if state.after_id:
        try:
            root.after_cancel(state.after_id)
        except Exception:
            pass
        state.after_id = None

    global camera_instance
    if camera_instance:
        try:
            camera_instance.stop()
        except:
            pass
        try:
            camera_instance.close()
        except:
            pass

        camera_instance = None

    reset_state()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    if root.winfo_exists():
        try:
            root.quit()
            root.destroy()  # ✅ protected against background callbacks
        except Exception:
            pass
        
    reset_state()
    state.gui_alive = True
    state.stop_requested = False

    launch_main_gui()

    
def compute_reflex_index(most_negative_slope):
    min_slope = -600
    max_slope = 0
    slope = max(min(most_negative_slope, max_slope), min_slope)
    return abs(slope) / abs(min_slope)

def plot_area_in_gui(area_list, time_list, workspace, container, root):
    for widget in container.winfo_children():
        widget.destroy()

    area = np.array(area_list)
    time_arr = np.array(time_list)
    record_duration_s = float(time_arr[-1]) if len(time_arr) > 0 else 0.0

    window_size = min(11, len(area) - 1)
    if window_size % 2 == 0:
        window_size -= 1
    smoothed_area = savgol_filter(area, window_size, polyorder=3) if window_size >= 3 else area
    slope_val = np.diff(smoothed_area) / np.diff(time_arr)

    segments = []
    current_segment = [0]
    most_negative_slope = 0

    for i in range(1, len(slope_val)):
        if (slope_val[i] > 0 and slope_val[i-1] < 0) or (slope_val[i] < 0 and slope_val[i-1] > 0):
            if len(current_segment) >= 50:
                segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)
    if len(current_segment) >= 50:
        segments.append(current_segment)

    colors = ['r', 'g', 'c', 'm', 'y']

    fatigue_msg = "FATIGUED"
    for idx, segment in enumerate(segments):
        t_seg = [time_arr[i+1] for i in segment]
        a_seg = [smoothed_area[i+1] for i in segment]
        if len(t_seg) > 1:
            slope, *_ = linregress(t_seg, a_seg)
            if slope < most_negative_slope:
                most_negative_slope = slope

    reflex_index = compute_reflex_index(most_negative_slope)
    if most_negative_slope < -400:
        fatigue_msg = "NOT FATIGUED"

    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    fig.subplots_adjust(left=0.5, right=0.95, top=0.88, bottom=0.4)
    #fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.15)

    ax.plot(time_arr, smoothed_area, color="black")
    for idx, segment in enumerate(segments):
        t_seg = [time_arr[i+1] for i in segment]
        a_seg = [smoothed_area[i+1] for i in segment]
        if len(t_seg) > 1:
            slope, *_ = linregress(t_seg, a_seg)
            ax.scatter(t_seg, a_seg, color=colors[idx % len(colors)], s=10)

    ax.set_title("Area vs Time", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=6)
    ax.set_ylabel("Area", fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(False)

    fig.text(0.05, 0.90, f"Status:\n{fatigue_msg}", fontsize=8, color="green" if "NOT" in fatigue_msg else "red",
             ha="left", va="top", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.6))
    fig.text(0.05, 0.35, f"RI: {reflex_index:.2f}", fontsize=8, ha="left", va="top",
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.4))
    fig.text(0.05, 0.20, "RI: 0 – 0.7 = FATIGUED\nRI: 0.7 - 1 = NOT FATIGUED", fontsize=8, ha="left", va="top",
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.5))

    fig.savefig(os.path.join(workspace, "area_vs_time.png"), dpi=150, bbox_inches="tight")
    
    
        
    summary = {
    "reflex_index": round(reflex_index, 3),
    "most_negative_slope": round(most_negative_slope, 3),
    "classification": "NOT_FATIGUED" if most_negative_slope < -400 else "FATIGUED",
    "record_duration_s": round(record_duration_s, 3),
    "num_samples": len(area_list),
    "analysis_timestamp": datetime.now().isoformat(timespec="seconds")
    }

    summary_path = os.path.join(workspace, "analysis_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    
    # --- Button Row ---
    btn_frame = ctk.CTkFrame(container)
    btn_frame.pack(fill="x", pady=10)

    back_btn = ctk.CTkButton(
        btn_frame,
        text="Back to Main Menu",
        width= 10,
        height=10,
        corner_radius=5,
        command=lambda: back_to_main(root)
    )
    back_btn.pack(side="left", padx=20)

    exit_btn = ctk.CTkButton(
        btn_frame,
        text="Exit",
        width= 10,
        height=10,
        corner_radius=5,
        command=lambda: (stop_leds(), setattr(state, 'stop_requested', True),
                         safe_close(root),
                         sys.exit())
    )
    exit_btn.pack(side="right", padx=50)

def run_ellipse_fitting_with_preview(video_label, container, root):
    from customtkinter import CTkImage
    frame_times = []
    frame_count = 0


    workspace = create_workspace()
    logger = setup_logger(workspace)
    area_csv_path = os.path.join(workspace, "area_raw.csv")


    global camera_instance
    camera_instance = Picamera2()
    camera_instance.preview_configuration.main.size = DISPLAY_RESOLUTION
    camera_instance.preview_configuration.main.format = "RGB888"
    camera_instance.start()
    
    recorded_frames = []

    area_list, time_list = [], []
    area_buffer = []
    BUFFER_SIZE = 20   # start with 20 (you can tune later)

    frame_idx = 0
    record_frame_idx = 0
    prev_ellipse = []
    detector = YoloPupilDetector()

    with open(area_csv_path, 'w') as f:
        f.write("time_s,frame_idx,area_px\n")


    # ===============================
    # GUI SAFE UPDATE FUNCTIONS
    # ===============================
    def gui_update_frame(pil_img):
        try:
            if not root.winfo_exists():
                return
            imgtk = CTkImage(light_image=pil_img, size=pil_img.size)
            video_label.configure(image=imgtk)
            video_label.image = imgtk
        except Exception as e:
            print("GUI frame update error:", e)

    def gui_finish():
        try:
            if not root.winfo_exists():
                return
            plot_area_in_gui(area_list, time_list, workspace, container, root)
        except Exception as e:
            print("GUI finish error:", e)

    # ===============================
    # WORKER LOOP (NO GUI CALLS)
    # ===============================
    fps_display = 0.0
    while True:
        
        try:
            elapsed = 0.0
            capture_time = time.time()
            
            if state.recording_active and state.recording_start_time is not None:
                elapsed = capture_time - state.recording_start_time
            
            # Stop condition
            if state.stop_requested or (
                state.recording_active and elapsed >= RECORD_DURATION
            ):
                break

            raw_frame = camera_instance.capture_array()
            # Picamera2 is configured as RGB888; OpenCV drawing expects BGR.
            frame_rgb = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # use this for processing & display
                
            if state.recording_active:
                record_frame_idx += 1

            # NOW it is safe
            if state.stop_requested or (
                state.recording_active and elapsed >= RECORD_DURATION
            ):
                break
            

            latest_area = 0.0
            latest_conf = 0.0

            roi_offset_x, roi_offset_y = 0, 0
            det_frame = frame
            prev_center_xy_det: Optional[Tuple[float, float]] = None
            if prev_ellipse:
                prev_center_xy_det = (float(prev_ellipse[-1][0]), float(prev_ellipse[-1][1]))

            if state.masking_enabled and prev_ellipse:
                cx, cy, size = prev_ellipse[-1]
                size = max(0, int(size))
                cx = int(cx)
                cy = int(cy)
                x1 = max(0, cx - size)
                y1 = max(0, cy - size)
                x2 = min(frame.shape[1], cx + size)
                y2 = min(frame.shape[0], cy + size)
                if x2 > x1 and y2 > y1:
                    roi_offset_x, roi_offset_y = x1, y1
                    det_frame = frame[y1:y2, x1:x2]
                    if prev_center_xy_det is not None:
                        prev_center_xy_det = (
                            prev_center_xy_det[0] - float(roi_offset_x),
                            prev_center_xy_det[1] - float(roi_offset_y),
                        )

            center_xy, axes_xy, area, _conf_best = detector.detect_best(
                det_frame,
                prev_center_xy=prev_center_xy_det,
            )
            if center_xy is not None and axes_xy is not None:
                center_full_x = float(center_xy[0]) + float(roi_offset_x)
                center_full_y = float(center_xy[1]) + float(roi_offset_y)

                axes1 = int(axes_xy[0])
                axes2 = int(axes_xy[1])
                cv2.ellipse(
                    frame,
                    (int(center_full_x), int(center_full_y)),
                    (axes1, axes2),
                    0,
                    0,
                    360,
                    (0, 255, 0),
                    2,
                )

                prev_ellipse.append((int(center_full_x), int(center_full_y), int(max(axes1, axes2)) * 2))
                latest_area = float(area)
                latest_conf = float(_conf_best)

                if state.recording_active:
                    area_buffer.append(
                        f"{elapsed:.3f},{record_frame_idx},{latest_area:.2f}\n"
                    )

                    if len(area_buffer) >= BUFFER_SIZE:
                        try:
                            with open(area_csv_path, 'a') as f:
                                f.writelines(area_buffer)
                            area_buffer.clear()
                        except Exception as e:
                            print("Buffer write error:", e)

                    area_list.append(latest_area)
                    time_list.append(elapsed)

            if len(prev_ellipse) > 3:
                # Keep the most recent detections for ROI cropping.
                prev_ellipse.pop(0)

            # Keep original frame for detection
            display_frame = cv2.resize(frame, DISPLAY_RESOLUTION)    

            if len(frame_times) >= 2:
                dt = frame_times[-1] - frame_times[-2]
                fps = 1.0 / dt if dt > 0 else 0.0
            else:
                fps = 0.0
            
            FPS_ALPHA = 0.9
            if frame_idx == 0:
                fps_display = fps
            else:
                fps_display = FPS_ALPHA * fps_display + (1 - FPS_ALPHA) * fps

            overlay = f"FPS: {fps_display:.1f} | Area: {latest_area:.1f} | Conf: {latest_conf:.2f}"

            cv2.putText(display_frame, overlay,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2)


            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            if state.recording_active:
                recorded_frames.append(frame.copy())
                frame_times.append(capture_time)
                frame_count += 1

            # THREAD SAFE GUI CALL
            if state.gui_alive and root.winfo_exists():
                state.after_id = root.after(0, gui_update_frame, pil_img)

            time.sleep(0.001)
            frame_idx += 1

        except Exception as e:
            print("Exception in worker loop:", e)
            break

    # ===============================
    # CLEANUP
    # ===============================
    try:
        # Final buffer flush
        if area_buffer:
            try:
                with open(area_csv_path, 'a') as f:
                    f.writelines(area_buffer)
            except Exception as e:
                print("Final buffer write error:", e)

        if camera_instance:
            try:
                camera_instance.stop()
            except:
                pass
            try:
                camera_instance.close()
            except:
                pass
            camera_instance = None

    except Exception as e:
        print("Camera cleanup error:", e)

    if not recorded_frames:
        # Happens if user never pressed "Record" before stopping/closing.
        logger.info("No recorded frames captured; skipping video output.")
    else:
        if len(frame_times) >= 2:
            start_time = frame_times[0]
            end_time = frame_times[-1]
            duration = end_time - start_time
            real_fps = frame_count / duration if duration > 0 else FRAME_RATE
        else:
            real_fps = FRAME_RATE

        h, w, _ = recorded_frames[0].shape

        video_writer = cv2.VideoWriter(
            os.path.join(workspace, "recorded_raw.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            real_fps,
            (w, h)
        )

        for f in recorded_frames:
            video_writer.write(f)

        video_writer.release()


    try:
        cv2.destroyAllWindows()
    except:
        pass

    logger.info("Recording ended.\n")

    # THREAD SAFE FINAL GUI CALL
    if state.gui_alive and root.winfo_exists():
        state.after_id = root.after(0, gui_finish)



def launch_processing_gui():
    root = ctk.CTk()
    root.title("Fatigue Analyzer")
    # Get actual screen resolution
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Force window to exactly match the TFT
    root.geometry(f"{screen_w}x{screen_h}+0+0")

    # Remove title bar & borders (CRITICAL for small TFTs)
    root.overrideredirect(True)

    # Optional: ensure it stays on top
    root.attributes("-topmost", True)
    
    
    ir_on_idle()
    #root.attributes("-fullscreen", True)


    def on_close():
        state.stop_requested = True
        state.gui_alive = False
        safe_close(root)
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_close)

    video_frame = ctk.CTkFrame(root)
    video_frame.pack(fill="both", expand=True)
    video_label = ctk.CTkLabel(video_frame, text="")
    video_label.pack(fill="both", expand=True)

    def start_recording():
        state.recording_active = True
        state.recording_start_time = time.time()

        start_leds()   # START LED SEQUENCE HERE

        record_btn.configure(state="disabled")
        stop_btn.configure(state="normal")

    record_btn = ctk.CTkButton(video_frame, text="Record", width=10, height=10, corner_radius=5, command=start_recording)
    record_btn.place(x=20, y=200)

    stop_btn = ctk.CTkButton(video_frame, text="Stop", width=10, height=10, corner_radius=5, command=lambda: (stop_leds(), ir_off(), setattr(state, 'stop_requested', True)) , state="disabled")
    stop_btn.place(x=175, y=200)

    #exit_btn = ctk.CTkButton(video_frame, text="Exit", width=10, height=10, corner_radius=5, command=lambda: (stop_leds(), setattr(state, 'stop_requested', True)) , safe_close(root), sys.exit())
    exit_btn = ctk.CTkButton(video_frame,text="Exit",width=10,height=10,corner_radius=5,command=lambda: (
        stop_leds(),ir_off(),
        setattr(state, 'stop_requested', True),
        safe_close(root),
        sys.exit()
    )

    exit_btn.place(x=230, y=200)

    threading.Thread(target=run_ellipse_fitting_with_preview, args=(video_label, video_frame, root), daemon=True).start()
    root.mainloop()

def launch_main_gui():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    main = ctk.CTk()
    main.title("Fatigue Analyzer")
    screen_w = main.winfo_screenwidth()
    screen_h = main.winfo_screenheight()

    main.geometry(f"{screen_w}x{screen_h}+0+0")
    main.overrideredirect(True)
    main.attributes("-topmost", True)
    #main.attributes("-fullscreen", True)


    label = ctk.CTkLabel(main, text="Fatigue Analyzer", font=("Arial", 20))
    label.pack(pady=40)

    def on_exit():
        main.destroy()
        sys.exit()

    def on_start():
        main.destroy()
        launch_processing_gui()

    def on_close():
        stop_leds()
        state.stop_requested = True
        main.after(500, main.destroy)
        sys.exit()

    main.protocol("WM_DELETE_WINDOW", on_close)

    btn_frame = ctk.CTkFrame(main)
    btn_frame.pack(side="bottom", pady=40)

    ctk.CTkButton(btn_frame, text="Start", command=on_start).pack(side="left", padx=10, pady=10)
    ctk.CTkButton(btn_frame, text="Exit", command=on_exit).pack(side="right", padx=10, pady=10)

    main.mainloop()

if __name__ == "__main__":
    launch_main_gui()





