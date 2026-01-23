#!/usr/bin/env python3
"""
SCRIPT 3: Efficient Person Follow (YOLO11 NCNN) + Lost Wait + Search Scan
========================================================================
Camera: 640x480
Alt target (safe): 5m

Behavior:
- TRACKING: follow person using BODY_NED vel + yaw_rate
- If target LOST:
    1) STOP/HOVER for 5 seconds (keep trying detection)
    2) SEARCH MODE:
        - Rotate LEFT 180° (body-relative) smoothly (yaw_rate)
        - If still not found, rotate back 180° to original direction
    3) If still lost, go back to LOST_WAIT and repeat.

No LOITER mode used.

Keys:
- q : quit
- s : stop (hover)
- l : land
"""

import time
import math
import threading
import cv2

from ultralytics import YOLO
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# =========================
# MODE (SITL / REAL)
# =========================
USE_SITL = True

SITL_CONNECTION = "tcp:127.0.0.1:5762"

REAL_CONNECTION = "/dev/ttyACM0"  # better: /dev/serial/by-id/XXXX
REAL_BAUD = 115200

# =========================
# CAMERA
# =========================
CAM_INDEX = 2
W, H = 640, 480
DISPLAY = True
CAM_FPS_LIMIT = 30
USE_MJPG = True

# =========================
# MODEL (NCNN)
# =========================
MODEL_PATH = "./yolo11n_ncnn_model"
PERSON_CLASS = 0

INFER_IMGSZ = 320
CONF = 0.35

# Hybrid inference (fast):
FULL_FRAME_EVERY = 10
ROI_MARGIN = 0.35
ROI_MIN_SIZE = 200

# =========================
# FLIGHT
# =========================
TARGET_ALTITUDE = 5.0

CONTROL_HZ = 15
TYPE_MASK_VEL_YAWRATE = 0x05C7  # use vel + yaw_rate, ignore pos/accel/yaw

MAX_VX = 1.0          # m/s
MAX_VY = 0.6          # m/s
MAX_YAW_RATE = 25.0   # deg/s

# Distance proxy via bbox height
TARGET_H_PX = 180

# Gains (start conservative)
Kp_yaw = 18.0
Kp_dist = 0.010
Kp_strafe = 0.40
USE_STRAFE = True

CMD_SMOOTH_ALPHA = 0.35
DEADBAND_X = 0.06
DEADBAND_H = 10

# =========================
# LOST + SEARCH MODE
# =========================
LOST_WAIT_SECONDS = 5.0     # wait (hover) before search
SEARCH_YAW_RATE = 20.0      # deg/s smooth scan
SEARCH_ANGLE = 180.0        # scan left 180 then return 180

# If yaw is reversed, flip:
YAW_SIGN = 1

# =========================
# UTILS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def now_s():
    return time.perf_counter()

# =========================
# CAMERA THREAD (LOW LATENCY)
# =========================
class CameraThread:
    def __init__(self, index=0, w=640, h=480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if USE_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Try CAM_INDEX=1 or check /dev/video0 permissions.")

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        last = now_s()
        min_dt = 1.0 / max(CAM_FPS_LIMIT, 1)

        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = f

            dt = now_s() - last
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last = now_s()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def close(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()

# =========================
# DRONE
# =========================
def connect_vehicle():
    if USE_SITL:
        v = connect(SITL_CONNECTION, wait_ready=True, timeout=60)
    else:
        v = connect(REAL_CONNECTION, baud=REAL_BAUD, wait_ready=True, timeout=60)
    print("Connected ✅", v.version)
    return v

def wait_until_armable(v):
    print("Waiting armable...")
    while not v.is_armable:
        time.sleep(1)
    print("Armable ✅")

def wait_for_gps(v):
    if not hasattr(v, "gps_0"):
        return
    print("Waiting GPS 3D fix...")
    while v.gps_0.fix_type < 3:
        print(f"GPS fix={v.gps_0.fix_type} sats={v.gps_0.satellites_visible}")
        time.sleep(1)
    print("GPS ✅")

def arm_and_takeoff(v, alt):
    print("Setting GUIDED...")
    v.mode = VehicleMode("GUIDED")
    while v.mode.name != "GUIDED":
        time.sleep(0.2)

    wait_until_armable(v)
    if not USE_SITL:
        wait_for_gps(v)

    print("Arming...")
    v.armed = True
    while not v.armed:
        time.sleep(0.2)

    print(f"Takeoff to {alt}m...")
    v.simple_takeoff(alt)
    while True:
        a = v.location.global_relative_frame.alt
        print(f"Alt: {a:.2f}m")
        if a >= 0.95 * alt:
            break
        time.sleep(1)

    time.sleep(2)
    print("Hover stable ✅")

def send_body_vel_yawrate(v, vx, vy, vz, yaw_rate_deg_s):
    yaw_rate_rad_s = math.radians(yaw_rate_deg_s) * YAW_SIGN
    msg = v.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate_rad_s
    )
    v.send_mavlink(msg)
    v.flush()

def stop_vehicle(v, seconds=1.0):
    end = time.time() + seconds
    dt = 1.0 / CONTROL_HZ
    while time.time() < end:
        send_body_vel_yawrate(v, 0, 0, 0, 0)
        time.sleep(dt)

def set_mode_safe(v, mode_name):
    try:
        v.mode = VehicleMode(mode_name)
    except Exception:
        pass

# =========================
# DETECTION
# =========================
def pick_largest_person(boxes):
    best = None
    best_area = 0.0
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else 0.0
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area > best_area:
            best_area = area
            best = (float(x1), float(y1), float(x2), float(y2), conf)
    return best

def crop_roi(frame, bbox, margin=0.35):
    x1, y1, x2, y2, _ = bbox
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin

    cx1 = int(max(0, x1 - mx))
    cy1 = int(max(0, y1 - my))
    cx2 = int(min(frame.shape[1] - 1, x2 + mx))
    cy2 = int(min(frame.shape[0] - 1, y2 + my))

    if (cx2 - cx1) < ROI_MIN_SIZE:
        pad = (ROI_MIN_SIZE - (cx2 - cx1)) // 2
        cx1 = max(0, cx1 - pad)
        cx2 = min(frame.shape[1] - 1, cx2 + pad)
    if (cy2 - cy1) < ROI_MIN_SIZE:
        pad = (ROI_MIN_SIZE - (cy2 - cy1)) // 2
        cy1 = max(0, cy1 - pad)
        cy2 = min(frame.shape[0] - 1, cy2 + pad)

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1, cx2, cy2)

# =========================
# UI OVERLAYS
# =========================
def draw_guides(frame, bbox_smooth, state_text, safe_alt_m, current_alt_m, err_x=None):
    # center lines + center point
    cx = W // 2
    cy = H // 2

    cv2.line(frame, (cx, 0), (cx, H), (180, 180, 180), 1)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    # Top-left text: safe distance (altitude), current altitude, state
    cv2.putText(frame, f"SAFE ALT: {safe_alt_m:.1f}m", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"ALT: {current_alt_m:.2f}m", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"STATE: {state_text}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Deviation line (center -> person center)
    if bbox_smooth is not None:
        x1, y1, x2, y2, cf = bbox_smooth
        px = int((x1 + x2) / 2)
        py = int((y1 + y2) / 2)

        cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)
        cv2.line(frame, (cx, cy), (px, py), (0, 0, 255), 2)

        if err_x is not None:
            cv2.putText(frame, f"err_x: {err_x:+.2f}", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

# =========================
# MAIN
# =========================
def main():
    cam = CameraThread(CAM_INDEX, W, H)
    model = YOLO(MODEL_PATH)

    vehicle = connect_vehicle()
    arm_and_takeoff(vehicle, TARGET_ALTITUDE)

    # States
    TRACKING = "TRACKING"
    LOST_WAIT = "LOST_WAIT_5S"
    SEARCH_LEFT = "SEARCH_LEFT_180"
    SEARCH_RETURN = "SEARCH_RETURN_180"

    state = TRACKING
    lost_start_t = None
    search_phase_start_t = None

    # for search timing
    search_phase_duration = abs(SEARCH_ANGLE / SEARCH_YAW_RATE)  # seconds for 180 at given rate

    # Target bbox (full frame)
    bbox_smooth = None

    # Smoothed commands
    vx_s = vy_s = yaw_s = 0.0

    det_count = 0
    prev_loop = now_s()
    fps_smooth = 0.0

    try:
        while True:
            # If user disarmed or changed mode, stop output
            if not vehicle.armed:
                stop_vehicle(vehicle, 1.0)
                break

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (W, H))

            # ---------- inference mode: full/ROI ----------
            use_full = (bbox_smooth is None) or (det_count % FULL_FRAME_EVERY == 0)
            roi_rect = None
            inp = frame
            if not use_full and bbox_smooth is not None:
                inp, roi_rect = crop_roi(frame, bbox_smooth, margin=ROI_MARGIN)

            t0 = now_s()
            results = model.predict(
                source=inp,
                imgsz=INFER_IMGSZ,
                conf=CONF,
                classes=[PERSON_CLASS],
                verbose=False
            )
            t1 = now_s()

            boxes = results[0].boxes
            found = None
            if boxes is not None and len(boxes) > 0:
                found = pick_largest_person(boxes)

            # map ROI -> full
            if found is not None and roi_rect is not None:
                ox1, oy1, _, _ = roi_rect
                x1, y1, x2, y2, cf = found
                found = (x1 + ox1, y1 + oy1, x2 + ox1, y2 + oy1, cf)

            # update bbox_smooth
            if found is not None:
                if bbox_smooth is None:
                    bbox_smooth = found
                else:
                    a = 0.35
                    x1 = (1-a)*bbox_smooth[0] + a*found[0]
                    y1 = (1-a)*bbox_smooth[1] + a*found[1]
                    x2 = (1-a)*bbox_smooth[2] + a*found[2]
                    y2 = (1-a)*bbox_smooth[3] + a*found[3]
                    cf = found[4]
                    bbox_smooth = (x1, y1, x2, y2, cf)

                # If found in any non-tracking state -> immediately back to tracking
                state = TRACKING
                lost_start_t = None
                search_phase_start_t = None

            # ---------- control ----------
            dt_control = 1.0 / CONTROL_HZ
            err_x_display = None

            if state == TRACKING:
                if bbox_smooth is None:
                    # if never found yet -> go to LOST_WAIT starting now
                    if lost_start_t is None:
                        lost_start_t = time.time()
                    state = LOST_WAIT
                    send_body_vel_yawrate(vehicle, 0, 0, 0, 0)
                else:
                    # compute control
                    x1, y1, x2, y2, cf = bbox_smooth
                    px = (x1 + x2) * 0.5
                    bh = max(1.0, (y2 - y1))

                    cx = W * 0.5
                    err_x = (px - cx) / cx  # [-1..1]
                    if abs(err_x) < DEADBAND_X:
                        err_x = 0.0
                    err_x_display = err_x

                    yaw_cmd = clamp(Kp_yaw * err_x, -MAX_YAW_RATE, MAX_YAW_RATE)

                    err_h = (TARGET_H_PX - bh)
                    if abs(err_h) < DEADBAND_H:
                        err_h = 0.0
                    vx_cmd = clamp(Kp_dist * err_h, -MAX_VX, MAX_VX)

                    if USE_STRAFE:
                        vy_cmd = clamp(Kp_strafe * err_x, -MAX_VY, MAX_VY)
                    else:
                        vy_cmd = 0.0

                    # smooth commands
                    a = CMD_SMOOTH_ALPHA
                    vx_s = (1-a)*vx_s + a*vx_cmd
                    vy_s = (1-a)*vy_s + a*vy_cmd
                    yaw_s = (1-a)*yaw_s + a*yaw_cmd

                    send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, yaw_s)

                    # If person disappears next loop, we’ll go LOST_WAIT
                    # (handled below if found is None and bbox becomes stale)

            # if not found in this loop -> treat as lost candidate
            if found is None:
                # If we were tracking, start LOST_WAIT timer
                if state == TRACKING:
                    state = LOST_WAIT
                    lost_start_t = time.time()
                    # hover immediately
                    send_body_vel_yawrate(vehicle, 0, 0, 0, 0)

            if state == LOST_WAIT:
                # hover for 5 seconds while still running detection
                send_body_vel_yawrate(vehicle, 0, 0, 0, 0)

                if lost_start_t is None:
                    lost_start_t = time.time()

                if (time.time() - lost_start_t) >= LOST_WAIT_SECONDS:
                    # enter search scan
                    state = SEARCH_LEFT
                    search_phase_start_t = time.time()
                    # clear smoothed commands so it doesn't jump
                    vx_s = vy_s = yaw_s = 0.0

            elif state == SEARCH_LEFT:
                # rotate LEFT (CCW) 180deg relative (body)
                # left = negative yaw_rate (if your yaw sign is opposite, flip YAW_SIGN)
                send_body_vel_yawrate(vehicle, 0, 0, 0, -SEARCH_YAW_RATE)

                if (time.time() - search_phase_start_t) >= search_phase_duration:
                    state = SEARCH_RETURN
                    search_phase_start_t = time.time()

            elif state == SEARCH_RETURN:
                # rotate back RIGHT (CW) 180deg
                send_body_vel_yawrate(vehicle, 0, 0, 0, +SEARCH_YAW_RATE)

                if (time.time() - search_phase_start_t) >= search_phase_duration:
                    # one scan cycle done -> go back to LOST_WAIT and wait 5s again
                    state = LOST_WAIT
                    lost_start_t = time.time()
                    send_body_vel_yawrate(vehicle, 0, 0, 0, 0)

            # ---------- HUD ----------
            det_count += 1
            infer_fps = 1.0 / max(t1 - t0, 1e-6)

            nowt = now_s()
            loop_fps = 1.0 / max(nowt - prev_loop, 1e-6)
            prev_loop = nowt
            fps_smooth = (0.9*fps_smooth + 0.1*loop_fps) if fps_smooth else loop_fps

            alt = vehicle.location.global_relative_frame.alt if vehicle.location.global_relative_frame else 0.0

            if DISPLAY:
                # draw bbox
                if bbox_smooth is not None:
                    x1, y1, x2, y2, cf = bbox_smooth
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(frame, f"conf={cf:.2f}", (int(x1), int(max(0, y1-8))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # guides + safe distance text
                draw_guides(frame, bbox_smooth, state, TARGET_ALTITUDE, alt, err_x_display)

                # FPS + commands
                cv2.putText(frame, f"LoopFPS:{fps_smooth:.1f}  InferFPS:{infer_fps:.1f}",
                            (10, H-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"vx:{vx_s:+.2f} vy:{vy_s:+.2f} yaw:{yaw_s:+.1f}  target_h:{TARGET_H_PX}",
                            (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.imshow("Person Follow (YOLO11 NCNN) - Search Mode", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    print("STOP")
                    stop_vehicle(vehicle, 1.0)
                elif key == ord("l"):
                    print("LAND")
                    set_mode_safe(vehicle, "LAND")
                    break

            time.sleep(dt_control)

    except KeyboardInterrupt:
        print("\nEMERGENCY -> LAND")
        set_mode_safe(vehicle, "LAND")

    finally:
        print("Stopping...")
        stop_vehicle(vehicle, 1.0)
        cam.close()
        if DISPLAY:
            cv2.destroyAllWindows()
        vehicle.close()
        print("Done ✅")

if __name__ == "__main__":
    main()
