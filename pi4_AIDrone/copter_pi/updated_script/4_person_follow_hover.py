#!/usr/bin/env python3
"""
SCRIPT 3: Dual-Camera Person Follow + Overhead Hover Mission (YOLO11 NCNN)
==========================================================================
Keeps SAME drone-control style (BODY_NED velocity + yaw_rate), same altitude, same max limits.
Only changes algorithm + dual-camera mission logic.

Cameras:
  - Front camera (CAM_FRONT): normal following
  - Down camera  (CAM_DOWN): overhead alignment + hover above head

Mission:
  1) Follow using front camera
  2) If person is stationary for 10s -> approach closer (front cam)
  3) When close + centered + lower in frame -> switch to down cam
  4) Align directly above person head (down cam) and hover 10-15s
  5) Switch back to front cam and back away (return), mission complete

Keys:
  q = quit
  s = STOP (hover)
  l = LAND
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
USE_SITL = False
SITL_CONNECTION = "tcp:127.0.0.1:5762"
REAL_CONNECTION = "/dev/ttyACM0"
REAL_BAUD = 115200

# =========================
# CAMERAS
# =========================
CAM_FRONT = 0
CAM_DOWN  = 2

W, H = 640, 480
DISPLAY = True
USE_MJPG = True
CAM_FPS_LIMIT = 30

# Down camera orientation correction (if needed)
# Options: None, "ROTATE_90_CW", "ROTATE_90_CCW", "ROTATE_180"
DOWN_ROTATE = None

# =========================
# MODEL (NCNN)
# =========================
MODEL_PATH = "./yolo11n_ncnn_model"
PERSON_CLASS = 0
INFER_IMGSZ = 320
CONF = 0.35

# Hybrid inference
FULL_FRAME_EVERY = 10
ROI_MARGIN = 0.35
ROI_MIN_SIZE = 200

# =========================
# FLIGHT (KEEP SAME)
# =========================
TARGET_ALTITUDE = 5.0

CONTROL_HZ = 15
TYPE_MASK_VEL_YAWRATE = 0x05C7  # use vel + yaw_rate; ignore pos/accel/yaw

MAX_VX = 1.0
MAX_VY = 0.6
MAX_YAW_RATE = 25.0

# Distance proxy via bbox height (front cam)
TARGET_H_PX = 180  # your normal "safe distance" proxy
# When approaching closer (still using front cam):
CLOSE_TARGET_H_PX = 240  # (tune) bigger => closer

# Gains (same spirit as before; keep conservative)
Kp_yaw = 18.0
Kp_dist = 0.010
Kp_strafe = 0.40
USE_STRAFE = True

CMD_SMOOTH_ALPHA = 0.35
DEADBAND_X = 0.06
DEADBAND_H = 10

# Down camera XY alignment (top-down)
# Map pixel error -> body velocities
Kp_down_xy = 0.55  # m/s per normalized error (clamped by MAX_VX/MAX_VY)
DOWN_DEADBAND = 0.05

# If yaw direction reversed, flip:
YAW_SIGN = 1

# =========================
# STATIONARY DETECTION (FRONT CAM)
# =========================
STOP_SECONDS = 10.0
STILL_CENTER_PX = 6.0       # center movement threshold (pixels per update)
STILL_SIZE_PX = 8.0         # bbox height change threshold

# =========================
# SWITCH CONDITIONS (FRONT -> DOWN)
# =========================
CENTER_X_TOL = 0.08          # must be centered horizontally
LOWER_Y_FRAC = 0.55          # person center y must be lower than this fraction of frame
CLOSE_H_TOL = 0.92           # bbox height must be >= CLOSE_TARGET_H_PX*tol
SWITCH_STABLE_SEC = 1.0      # stable for this long before switching

# =========================
# OVERHEAD HOVER
# =========================
ALIGN_TOL = 0.06             # down cam alignment tolerance
ALIGN_STABLE_SEC = 1.5       # hold centered before starting hover timer
OVERHEAD_HOVER_SEC = 12.0    # 10-15 sec (set 10..15)

# =========================
# RETURN
# =========================
RETURN_BACK_SEC = 5.0
RETURN_VX = -0.8  # move backward (clamped)

# =========================
# LOST TARGET BEHAVIOR (KEEP YOUR WORKING STYLE)
# =========================
LOST_WAIT_SECONDS = 5.0
SEARCH_YAW_RATE = 20.0
SEARCH_ANGLE = 180.0

# =========================
# UTILS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rotate_frame(frame, mode):
    if mode is None:
        return frame
    if mode == "ROTATE_90_CW":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ROTATE_90_CCW":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "ROTATE_180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

# =========================
# CAMERA THREAD
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
            raise RuntimeError(f"Could not open camera index={index}")

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        last = time.perf_counter()
        min_dt = 1.0 / max(CAM_FPS_LIMIT, 1)

        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

            dt = time.perf_counter() - last
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last = time.perf_counter()

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
        0, 0, 0,
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
# UI
# =========================
def draw_ui(frame, bbox, state, active_cam, alt, vx, vy, yaw, extra=""):
    cx = W // 2
    cy = H // 2
    cv2.line(frame, (cx, 0), (cx, H), (180,180,180), 1)
    cv2.circle(frame, (cx, cy), 4, (255,255,255), -1)

    if bbox is not None:
        x1,y1,x2,y2,cf = bbox
        px = int((x1+x2)/2)
        py = int((y1+y2)/2)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.circle(frame, (px,py), 4, (255,0,0), -1)
        cv2.line(frame, (cx,cy), (px,py), (0,0,255), 2)

    # top-left info
    cv2.putText(frame, f"SAFE ALT: {TARGET_ALTITUDE:.1f}m  ALT:{alt:.2f}m", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"STATE: {state}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"CAM: {active_cam}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    if extra:
        cv2.putText(frame, extra, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # bottom info
    cv2.putText(frame, f"vx:{vx:+.2f} vy:{vy:+.2f} yaw:{yaw:+.1f}", (10, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# =========================
# MAIN
# =========================
def main():
    cam_front = CameraThread(CAM_FRONT, W, H)
    cam_down  = CameraThread(CAM_DOWN,  W, H)

    model = YOLO(MODEL_PATH)

    vehicle = connect_vehicle()
    arm_and_takeoff(vehicle, TARGET_ALTITUDE)

    # STATES
    FOLLOW_FRONT   = "FOLLOW_FRONT"
    APPROACH_FRONT = "APPROACH_FRONT"
    OVERHEAD_ALIGN = "OVERHEAD_ALIGN_DOWN"
    OVERHEAD_HOVER = "OVERHEAD_HOVER_10_15S"
    RETURN_BACK    = "RETURN_BACK"
    DONE           = "DONE"

    LOST_WAIT      = "LOST_WAIT_5S"
    SEARCH_LEFT    = "SEARCH_LEFT_180"
    SEARCH_RETURN  = "SEARCH_RETURN_180"

    state = FOLLOW_FRONT
    resume_state = FOLLOW_FRONT  # after reacquire

    # timers
    stationary_t = 0.0
    last_bbox_for_stop = None
    last_bbox_time = None

    lost_start_t = None
    search_start_t = None
    search_duration = abs(SEARCH_ANGLE / SEARCH_YAW_RATE)

    switch_stable_start = None
    align_stable_start = None
    hover_start = None
    return_start = None

    # bbox smoothing per active cam
    bbox_front = None
    bbox_down  = None

    # command smoothing
    vx_s = vy_s = yaw_s = 0.0

    det_count = 0

    def run_detect(frame, bbox_smooth):
        nonlocal det_count
        use_full = (bbox_smooth is None) or (det_count % FULL_FRAME_EVERY == 0)
        roi_rect = None
        inp = frame
        if not use_full and bbox_smooth is not None:
            inp, roi_rect = crop_roi(frame, bbox_smooth, margin=ROI_MARGIN)

        results = model.predict(
            source=inp,
            imgsz=INFER_IMGSZ,
            conf=CONF,
            classes=[PERSON_CLASS],
            verbose=False
        )
        boxes = results[0].boxes
        found = None
        if boxes is not None and len(boxes) > 0:
            found = pick_largest_person(boxes)

        # map ROI -> full
        if found is not None and roi_rect is not None:
            ox1, oy1, _, _ = roi_rect
            x1,y1,x2,y2,cf = found
            found = (x1+ox1, y1+oy1, x2+ox1, y2+oy1, cf)

        det_count += 1
        return found

    def ema_bbox(old, new, a=0.35):
        if new is None:
            return old
        if old is None:
            return new
        x1 = (1-a)*old[0] + a*new[0]
        y1 = (1-a)*old[1] + a*new[1]
        x2 = (1-a)*old[2] + a*new[2]
        y2 = (1-a)*old[3] + a*new[3]
        cf = new[4]
        return (x1,y1,x2,y2,cf)

    try:
        while True:
            if not vehicle.armed:
                stop_vehicle(vehicle, 1.0)
                break

            # choose active camera based on state
            if state in [OVERHEAD_ALIGN, OVERHEAD_HOVER]:
                raw = cam_down.read()
                active_cam_name = "DOWN"
                if raw is None:
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(raw, (W,H))
                frame = rotate_frame(frame, DOWN_ROTATE)
                found = run_detect(frame, bbox_down)
                bbox_down = ema_bbox(bbox_down, found)
                bbox_active = bbox_down
            else:
                raw = cam_front.read()
                active_cam_name = "FRONT"
                if raw is None:
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(raw, (W,H))
                found = run_detect(frame, bbox_front)
                bbox_front = ema_bbox(bbox_front, found)
                bbox_active = bbox_front

            alt = vehicle.location.global_relative_frame.alt if vehicle.location.global_relative_frame else 0.0

            # -------------------------------------------------------
            # LOST handling (global) — keep your working WAIT+SEARCH
            # -------------------------------------------------------
            if bbox_active is None:
                # enter LOST_WAIT if not already in lost/search
                if state not in [LOST_WAIT, SEARCH_LEFT, SEARCH_RETURN]:
                    resume_state = state
                    state = LOST_WAIT
                    lost_start_t = time.time()
                    search_start_t = None
                    stop_vehicle(vehicle, 0.2)

            # Lost states
            if state == LOST_WAIT:
                # hover for 5 seconds while still detecting
                send_body_vel_yawrate(vehicle, 0, 0, 0, 0)
                if (time.time() - lost_start_t) >= LOST_WAIT_SECONDS:
                    state = SEARCH_LEFT
                    search_start_t = time.time()
                # if target appears again, resume
                if bbox_active is not None:
                    state = resume_state
                    lost_start_t = None
                    search_start_t = None

            elif state == SEARCH_LEFT:
                # rotate left 180
                send_body_vel_yawrate(vehicle, 0, 0, 0, -SEARCH_YAW_RATE)
                if (time.time() - search_start_t) >= search_duration:
                    state = SEARCH_RETURN
                    search_start_t = time.time()
                if bbox_active is not None:
                    state = resume_state
                    lost_start_t = None
                    search_start_t = None

            elif state == SEARCH_RETURN:
                # rotate back 180
                send_body_vel_yawrate(vehicle, 0, 0, 0, +SEARCH_YAW_RATE)
                if (time.time() - search_start_t) >= search_duration:
                    # cycle again
                    state = LOST_WAIT
                    lost_start_t = time.time()
                    search_start_t = None
                if bbox_active is not None:
                    state = resume_state
                    lost_start_t = None
                    search_start_t = None

            # -------------------------------------------------------
            # Mission states
            # -------------------------------------------------------
            if state in [FOLLOW_FRONT, APPROACH_FRONT] and bbox_front is not None:
                # FRONT control (same controller, only target height changes during approach)
                x1,y1,x2,y2,cf = bbox_front
                px = (x1+x2) * 0.5
                py = (y1+y2) * 0.5
                bh = max(1.0, (y2-y1))

                cx = W * 0.5
                cy = H * 0.5
                err_x = (px - cx) / cx  # [-1..1]
                if abs(err_x) < DEADBAND_X:
                    err_x = 0.0

                yaw_cmd = clamp(Kp_yaw * err_x, -MAX_YAW_RATE, MAX_YAW_RATE)

                # normal follow vs approach
                if state == FOLLOW_FRONT:
                    target_h = TARGET_H_PX
                else:
                    target_h = CLOSE_TARGET_H_PX

                err_h = (target_h - bh)
                if abs(err_h) < DEADBAND_H:
                    err_h = 0.0
                vx_cmd = clamp(Kp_dist * err_h, -MAX_VX, MAX_VX)

                if USE_STRAFE:
                    vy_cmd = clamp(Kp_strafe * err_x, -MAX_VY, MAX_VY)
                else:
                    vy_cmd = 0.0

                # smooth
                a = CMD_SMOOTH_ALPHA
                vx_s = (1-a)*vx_s + a*vx_cmd
                vy_s = (1-a)*vy_s + a*vy_cmd
                yaw_s = (1-a)*yaw_s + a*yaw_cmd

                # send commands
                send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, yaw_s)

                # ---------------------------
                # STOP detection (only in FOLLOW_FRONT)
                # ---------------------------
                if state == FOLLOW_FRONT:
                    now = time.time()
                    if last_bbox_for_stop is not None and last_bbox_time is not None:
                        px_prev = (last_bbox_for_stop[0] + last_bbox_for_stop[2]) * 0.5
                        bh_prev = max(1.0, (last_bbox_for_stop[3] - last_bbox_for_stop[1]))
                        d_px = abs(px - px_prev)
                        d_bh = abs(bh - bh_prev)

                        if d_px < STILL_CENTER_PX and d_bh < STILL_SIZE_PX:
                            stationary_t += (now - last_bbox_time)
                        else:
                            stationary_t = 0.0

                    last_bbox_for_stop = (x1,y1,x2,y2,cf)
                    last_bbox_time = now

                    # if person stopped for 10s -> approach
                    if stationary_t >= STOP_SECONDS:
                        state = APPROACH_FRONT
                        switch_stable_start = None

                # ---------------------------
                # Switch condition (only in APPROACH_FRONT)
                # ---------------------------
                if state == APPROACH_FRONT:
                    # criteria: centered X + lower in frame + close height
                    err_x_abs = abs((px - cx) / cx)
                    lower_ok = (py > H * LOWER_Y_FRAC)
                    close_ok = (bh >= CLOSE_TARGET_H_PX * CLOSE_H_TOL)
                    center_ok = (err_x_abs <= CENTER_X_TOL)

                    if center_ok and lower_ok and close_ok:
                        if switch_stable_start is None:
                            switch_stable_start = time.time()
                        elif (time.time() - switch_stable_start) >= SWITCH_STABLE_SEC:
                            # Switch to down camera
                            state = OVERHEAD_ALIGN
                            align_stable_start = None
                            hover_start = None
                            # reset command smoothing to avoid jump
                            vx_s = vy_s = yaw_s = 0.0
                    else:
                        switch_stable_start = None

            # DOWN camera alignment + hover
            if state in [OVERHEAD_ALIGN, OVERHEAD_HOVER] and bbox_down is not None:
                x1,y1,x2,y2,cf = bbox_down
                px = (x1+x2) * 0.5
                py = (y1+y2) * 0.5

                cx = W * 0.5
                cy = H * 0.5
                err_x = (px - cx) / cx   # right is +
                err_y = (py - cy) / cy   # down is +

                # deadband
                if abs(err_x) < DOWN_DEADBAND:
                    err_x = 0.0
                if abs(err_y) < DOWN_DEADBAND:
                    err_y = 0.0

                # Map down-cam error to body motion:
                # If person is above center (err_y negative), move forward (+vx) to bring them to center.
                vx_cmd = clamp(-Kp_down_xy * err_y, -MAX_VX, MAX_VX)
                vy_cmd = clamp(+Kp_down_xy * err_x, -MAX_VY, MAX_VY)

                # keep yaw_rate 0 for overhead (stable)
                yaw_cmd = 0.0

                # smooth
                a = CMD_SMOOTH_ALPHA
                vx_s = (1-a)*vx_s + a*vx_cmd
                vy_s = (1-a)*vy_s + a*vy_cmd
                yaw_s = (1-a)*yaw_s + a*yaw_cmd

                send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, yaw_s)

                aligned = (abs(err_x) <= ALIGN_TOL) and (abs(err_y) <= ALIGN_TOL)

                if state == OVERHEAD_ALIGN:
                    if aligned:
                        if align_stable_start is None:
                            align_stable_start = time.time()
                        elif (time.time() - align_stable_start) >= ALIGN_STABLE_SEC:
                            state = OVERHEAD_HOVER
                            hover_start = time.time()
                    else:
                        align_stable_start = None

                elif state == OVERHEAD_HOVER:
                    # keep minor corrections, and count hover time
                    if hover_start is not None and (time.time() - hover_start) >= OVERHEAD_HOVER_SEC:
                        state = RETURN_BACK
                        return_start = time.time()
                        # switch back to front camera logic
                        vx_s = vy_s = yaw_s = 0.0

            # RETURN_BACK
            if state == RETURN_BACK:
                # move backward with front camera active (no detection needed)
                send_body_vel_yawrate(vehicle, clamp(RETURN_VX, -MAX_VX, MAX_VX), 0.0, 0.0, 0.0)
                if (time.time() - return_start) >= RETURN_BACK_SEC:
                    stop_vehicle(vehicle, 1.0)
                    state = DONE

            if state == DONE:
                # mission completed (hover)
                send_body_vel_yawrate(vehicle, 0, 0, 0, 0)

            # UI
            if DISPLAY:
                extra = ""
                if state == FOLLOW_FRONT:
                    extra = f"StoppedTimer: {stationary_t:.1f}/{STOP_SECONDS}s"
                if state == APPROACH_FRONT:
                    extra = f"Approaching... targetH={CLOSE_TARGET_H_PX}px"
                if state == OVERHEAD_HOVER and hover_start is not None:
                    extra = f"OverheadHover: {time.time()-hover_start:.1f}/{OVERHEAD_HOVER_SEC}s"

                draw_ui(frame, bbox_active, state, active_cam_name, alt, vx_s, vy_s, yaw_s, extra)
                cv2.imshow("DualCam Follow + Overhead", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    stop_vehicle(vehicle, 1.0)
                elif key == ord("l"):
                    print("LAND")
                    set_mode_safe(vehicle, "LAND")
                    break

            time.sleep(1.0 / CONTROL_HZ)

    except KeyboardInterrupt:
        print("\nEMERGENCY -> LAND")
        set_mode_safe(vehicle, "LAND")

    finally:
        print("Stopping...")
        stop_vehicle(vehicle, 1.0)
        cam_front.close()
        cam_down.close()
        if DISPLAY:
            cv2.destroyAllWindows()
        vehicle.close()
        print("Done ✅")

if __name__ == "__main__":
    main()
