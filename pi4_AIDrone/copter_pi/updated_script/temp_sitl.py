#!/usr/bin/env python3
"""
Dual-camera YOLO Person Follow (SITL + Real) — ACTIVE FEED ONLY
==============================================================

BEHAVIOR:
1) Person stops for 5 seconds on FRONT:
   -> FRONT_APPROACH (move toward person)
   -> if close+center -> switch to DOWN camera and keep following

2) LOST handling:
   -> If target disappears in general: drone STOPS, waits 3 seconds, then shuffles camera once.

✅ NEW CHANGE (requested):
- During FRONT_APPROACH, if person is NOT visible on FRONT,
  switch IMMEDIATELY to DOWN (no 3 second wait).
- 3 seconds applies ONLY when there is no person (general lost case).
"""

import time, math, threading, sys
import cv2
from ultralytics import YOLO
from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ============================================================
# CONFIG
# ============================================================

# =========================
# MODE (SITL / REAL)
# =========================
USE_SITL = False
SITL_CONNECTION = "tcp:127.0.0.1:5762"
REAL_CONNECTION = "/dev/ttyACM0"
REAL_BAUD = 115200

# Cameras
CAM_FRONT = 2
CAM_DOWN  = 0

W, H = 640, 480
DISPLAY = True
USE_MJPG = True
CAM_FPS_LIMIT = 30
DOWN_ROTATE = None  # None / "ROTATE_90_CW" / "ROTATE_90_CCW" / "ROTATE_180"

# Model
MODEL_PATH = "yolo11n_ncnn_model"
PERSON_CLASS = 0
INFER_IMGSZ = 320
CONF = 0.35

# ROI acceleration
FULL_FRAME_EVERY = 10
ROI_MARGIN = 0.35
ROI_MIN_SIZE = 200

# Robust tracking
BBOX_MISS_MAX = 10
BBOX_STALE_SEC = 1.2

# Flight/Control
AUTO_TAKEOFF = True
TARGET_ALTITUDE = 5.0
CONTROL_HZ = 15

TYPE_MASK_VEL_YAWRATE = 0x05C7
MAX_VX = 1.0
MAX_VY = 0.6
MAX_YAW_RATE = 25.0

TARGET_H_PX = 180
CLOSE_TARGET_H_PX = 240

Kp_yaw = 18.0
Kp_dist = 0.010
Kp_strafe = 0.40
USE_STRAFE = True

CMD_SMOOTH_ALPHA = 0.35
DEADBAND_X = 0.06
DEADBAND_H = 10

# DOWN follow gains
Kp_down_xy = 0.55
DOWN_DEADBAND = 0.05

YAW_SIGN = 1

# stop-detect -> approach
STOP_SECONDS = 5.0
STILL_CENTER_PX = 6.0
STILL_SIZE_PX = 8.0

# switch FRONT->DOWN condition
CENTER_X_TOL = 0.08
CLOSE_H_TOL = 0.92
SWITCH_STABLE_SEC = 1.0  # keep if you want stability; switch is already immediate if FRONT loses person

# ✅ Only for "no person" (general lost)
CAM_SHUFFLE_LOST_SEC = 3.0

# after switching camera: force full-frame detect briefly
FORCE_FULL_AFTER_SWITCH_SEC = 1.0


# ============================================================
# Helpers
# ============================================================

def clamp(x, lo, hi): return max(lo, min(hi, x))

def rotate_frame(frame, mode):
    if mode is None: return frame
    if mode == "ROTATE_90_CW": return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ROTATE_90_CCW": return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "ROTATE_180": return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def open_capture(source, w, h, prefer_mjpg=True):
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {source}")
        return cap

    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    else:
        backends = [cv2.CAP_V4L2, 0]

    for be in backends:
        cap = cv2.VideoCapture(source, be) if be != 0 else cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if cap.isOpened():
            return cap
        cap.release()

    raise RuntimeError(f"Could not open camera index={source}")

class CameraThread:
    def __init__(self, source, w, h, name="cam"):
        self.name = name
        self.cap = open_capture(source, w, h, prefer_mjpg=USE_MJPG)
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
        try:
            self.cap.release()
        except Exception:
            pass


# ============================================================
# Drone helpers
# ============================================================

def connect_vehicle():
    if USE_SITL:
        v = connect(SITL_CONNECTION, wait_ready=True, timeout=60)
    else:
        v = connect(REAL_CONNECTION, baud=REAL_BAUD, wait_ready=True, timeout=60)
    print("Connected ✅", v.version)
    return v

def wait_until_armable(v):
    while not v.is_armable:
        time.sleep(1)

def set_mode_wait(v, mode_name, timeout=10):
    v.mode = VehicleMode(mode_name)
    t0 = time.time()
    while v.mode.name != mode_name and (time.time() - t0) < timeout:
        time.sleep(0.2)
    return (v.mode.name == mode_name)

def arm_and_takeoff(v, alt):
    ok = set_mode_wait(v, "GUIDED", timeout=15)
    if not ok:
        raise RuntimeError("Failed to enter GUIDED mode")

    wait_until_armable(v)

    print("[ARM] Arming...")
    v.armed = True
    while not v.armed:
        time.sleep(0.2)

    print(f"[TAKEOFF] Taking off to {alt:.1f}m...")
    v.simple_takeoff(alt)

    while True:
        a = float(v.location.global_relative_frame.alt or 0.0)
        if a >= 0.95 * alt:
            break
        time.sleep(0.5)

    print("[TAKEOFF] Altitude reached ✅")
    time.sleep(1.0)

def send_body_vel_yawrate(v, vx, vy, vz, yaw_rate_deg_s):
    yaw_rate_rad_s = math.radians(yaw_rate_deg_s) * YAW_SIGN
    msg = v.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK_VEL_YAWRATE,
        0,0,0,
        vx,vy,vz,
        0,0,0,
        0, yaw_rate_rad_s
    )
    v.send_mavlink(msg)
    v.flush()

def stop_vehicle(v, seconds=0.2):
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


# ============================================================
# Detection + Tracking
# ============================================================

def pick_largest_person(boxes):
    best, best_area = None, 0.0
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else 0.0
        area = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
        if area > best_area:
            best_area = area
            best = (float(x1), float(y1), float(x2), float(y2), conf)
    return best

def crop_roi(frame, bbox, margin=0.35):
    h, w = frame.shape[:2]
    x1,y1,x2,y2,_ = bbox
    bw, bh = x2-x1, y2-y1
    mx, my = bw*margin, bh*margin

    cx1 = int(max(0, x1-mx)); cy1 = int(max(0, y1-my))
    cx2 = int(min(w, x2+mx));  cy2 = int(min(h, y2+my))

    if (cx2-cx1) < ROI_MIN_SIZE:
        pad = (ROI_MIN_SIZE-(cx2-cx1))//2
        cx1 = max(0, cx1-pad)
        cx2 = min(w, cx2+pad)
    if (cy2-cy1) < ROI_MIN_SIZE:
        pad = (ROI_MIN_SIZE-(cy2-cy1))//2
        cy1 = max(0, cy1-pad)
        cy2 = min(h, cy2+pad)

    cx1 = int(clamp(cx1, 0, w-1))
    cy1 = int(clamp(cy1, 0, h-1))
    cx2 = int(clamp(cx2, cx1+1, w))
    cy2 = int(clamp(cy2, cy1+1, h))

    roi = frame[cy1:cy2, cx1:cx2]
    return roi, (cx1, cy1, cx2, cy2)

def ema_bbox(old, new, a=0.35):
    if old is None: return new
    if new is None: return None
    x1=(1-a)*old[0]+a*new[0]; y1=(1-a)*old[1]+a*new[1]
    x2=(1-a)*old[2]+a*new[2]; y2=(1-a)*old[3]+a*new[3]
    return (x1,y1,x2,y2,new[4])

def update_track(bbox_smooth, miss_count, last_seen_t, found, alpha=CMD_SMOOTH_ALPHA):
    now = time.time()
    if found is None:
        miss_count += 1
        if last_seen_t and (now - last_seen_t) > BBOX_STALE_SEC:
            miss_count = BBOX_MISS_MAX
        if miss_count >= BBOX_MISS_MAX:
            return None, miss_count, last_seen_t
        return bbox_smooth, miss_count, last_seen_t

    last_seen_t = now
    miss_count = 0
    bbox_smooth = ema_bbox(bbox_smooth, found, alpha)
    return bbox_smooth, miss_count, last_seen_t

def detect(model, frame, bbox_hint, det_count, force_full=False):
    if frame is None or frame.size == 0:
        return None, det_count + 1

    use_full = force_full or (bbox_hint is None) or (det_count % FULL_FRAME_EVERY == 0)
    roi_rect = None
    inp = frame

    if not use_full and bbox_hint is not None:
        inp, roi_rect = crop_roi(frame, bbox_hint, ROI_MARGIN)
        if inp is None or inp.size == 0:
            return None, det_count + 1

    r = model.predict(inp, imgsz=INFER_IMGSZ, conf=CONF, classes=[PERSON_CLASS], verbose=False)
    boxes = r[0].boxes if r and len(r) > 0 else None

    found = None
    if boxes is not None and len(boxes) > 0:
        found = pick_largest_person(boxes)

    if found is not None and roi_rect is not None:
        ox1, oy1, _, _ = roi_rect
        x1, y1, x2, y2, cf = found
        found = (x1+ox1, y1+oy1, x2+ox1, y2+oy1, cf)

    return found, det_count + 1


# ============================================================
# UI
# ============================================================

def draw_overlay(frame, bbox, cam_label, mode, alt, vx, vy, yaw, extra=""):
    h, w = frame.shape[:2]
    cx, cy = int(w/2), int(h/2)

    cv2.line(frame, (cx, 0), (cx, h), (180,180,180), 1)
    cv2.line(frame, (0, cy), (w, cy), (180,180,180), 1)
    cv2.circle(frame, (cx, cy), 4, (255,255,255), -1)

    if bbox:
        x1,y1,x2,y2,cf = bbox
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"conf:{cf:.2f}", (int(x1), max(0,int(y1)-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(frame, f"ACTIVE CAM: {cam_label}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
    cv2.putText(frame, f"MODE: {mode}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    if extra:
        cv2.putText(frame, extra, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,255), 2)

    cv2.putText(frame, f"ALT:{alt:.2f}  vx:{vx:+.2f} vy:{vy:+.2f} yaw:{yaw:+.1f}",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)


# ============================================================
# MAIN
# ============================================================

def main():
    cam_front = None
    cam_down  = None
    vehicle   = None

    MODE_FRONT_FOLLOW   = "FRONT_FOLLOW"
    MODE_FRONT_APPROACH = "FRONT_APPROACH"
    MODE_DOWN_FOLLOW    = "DOWN_FOLLOW"

    mode = MODE_FRONT_FOLLOW
    active_cam = "FRONT"

    bbox_front = None
    bbox_down  = None
    miss_front = 0
    miss_down  = 0
    last_seen_front = 0.0
    last_seen_down  = 0.0
    det_count = 0

    vx_s = vy_s = yaw_s = 0.0

    stationary_t = 0.0
    last_bbox_for_stop = None
    last_bbox_time = None

    switch_stable_start = None

    # general lost timer (3 seconds)
    missing_since = None
    force_full_until = 0.0

    def set_active_cam(new_cam):
        nonlocal active_cam, missing_since, force_full_until, vx_s, vy_s, yaw_s
        active_cam = new_cam
        missing_since = None
        force_full_until = time.time() + FORCE_FULL_AFTER_SWITCH_SEC
        vx_s = vy_s = yaw_s = 0.0
        print(f"[CAM] -> {active_cam}")

    try:
        cam_front = CameraThread(CAM_FRONT, W, H, "front")
        cam_down  = CameraThread(CAM_DOWN,  W, H, "down") if CAM_DOWN != CAM_FRONT else cam_front

        print("[MODEL] Loading:", MODEL_PATH)
        model = YOLO(MODEL_PATH)
        print("[MODEL] Loaded ✅")

        vehicle = connect_vehicle()

        if DISPLAY:
            cv2.namedWindow("ACTIVE Camera (DualCam Follow)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ACTIVE Camera (DualCam Follow)", W, H)

        if AUTO_TAKEOFF:
            arm_and_takeoff(vehicle, TARGET_ALTITUDE)

        while True:
            raw_f = cam_front.read()
            raw_d = cam_down.read() if cam_down else None

            frame_front = cv2.resize(raw_f, (W, H)) if raw_f is not None else None
            frame_down = None
            if raw_d is not None:
                frame_down = cv2.resize(raw_d, (W, H))
                frame_down = rotate_frame(frame_down, DOWN_ROTATE)

            alt = 0.0
            try:
                alt = float(vehicle.location.global_relative_frame.alt or 0.0)
            except Exception:
                pass

            # Pick active frame + bbox hint
            if active_cam == "FRONT":
                active_frame = frame_front
                bbox_hint = bbox_front
            else:
                active_frame = frame_down
                bbox_hint = bbox_down

            if active_frame is None:
                time.sleep(0.01)
                continue

            # Detect on active cam
            force_full = (time.time() < force_full_until)
            found, det_count = detect(model, active_frame, bbox_hint, det_count, force_full=force_full)

            # Update active track
            if active_cam == "FRONT":
                bbox_front, miss_front, last_seen_front = update_track(bbox_front, miss_front, last_seen_front, found)
                bbox_active = bbox_front
                miss_active = miss_front
            else:
                bbox_down, miss_down, last_seen_down = update_track(bbox_down, miss_down, last_seen_down, found)
                bbox_active = bbox_down
                miss_active = miss_down

            # ============================================================
            # ✅ NEW: If approaching and FRONT loses person -> switch immediately to DOWN
            # ============================================================
            if mode == MODE_FRONT_APPROACH and active_cam == "FRONT" and bbox_active is None:
                stop_vehicle(vehicle, 0.10)
                set_active_cam("DOWN")
                mode = MODE_DOWN_FOLLOW
                # important: continue so next loop detects on DOWN before any control
                continue

            # ============================================================
            # General LOST handling (3 seconds ONLY when no person)
            # ============================================================
            if bbox_active is None:
                stop_vehicle(vehicle, 0.15)

                if missing_since is None:
                    missing_since = time.time()
                else:
                    if (time.time() - missing_since) >= CAM_SHUFFLE_LOST_SEC:
                        set_active_cam("DOWN" if active_cam == "FRONT" else "FRONT")
                        mode = MODE_FRONT_FOLLOW if active_cam == "FRONT" else MODE_DOWN_FOLLOW
                        switch_stable_start = None
                        missing_since = None

                if DISPLAY:
                    disp = active_frame.copy()
                    remain = CAM_SHUFFLE_LOST_SEC - (time.time() - missing_since) if missing_since else CAM_SHUFFLE_LOST_SEC
                    extra = f"NO PERSON | miss:{miss_active} | shuffle in {max(0.0, remain):.1f}s"
                    draw_overlay(disp, None, active_cam, mode, alt, 0.0, 0.0, 0.0, extra)
                    cv2.imshow("ACTIVE Camera (DualCam Follow)", disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    if k == ord('l'):
                        set_mode_safe(vehicle, "LAND")
                        break

                time.sleep(1.0 / CONTROL_HZ)
                continue
            else:
                missing_since = None

            # ============================================================
            # CONTROL
            # ============================================================

            if mode in [MODE_FRONT_FOLLOW, MODE_FRONT_APPROACH]:
                if active_cam != "FRONT":
                    mode = MODE_DOWN_FOLLOW
                else:
                    x1,y1,x2,y2,cf = bbox_front
                    px = (x1+x2)*0.5
                    py = (y1+y2)*0.5
                    bh = max(1.0, (y2-y1))

                    fh, fw = frame_front.shape[:2]
                    cx = fw*0.5

                    err_x = (px - cx) / cx
                    if abs(err_x) < DEADBAND_X:
                        err_x = 0.0

                    yaw_cmd = clamp(Kp_yaw * err_x, -MAX_YAW_RATE, MAX_YAW_RATE)

                    target_h = TARGET_H_PX if mode == MODE_FRONT_FOLLOW else CLOSE_TARGET_H_PX
                    err_h = (target_h - bh)
                    if abs(err_h) < DEADBAND_H:
                        err_h = 0.0

                    vx_cmd = clamp(Kp_dist * err_h, -MAX_VX, MAX_VX)
                    vy_cmd = clamp(Kp_strafe * err_x, -MAX_VY, MAX_VY) if USE_STRAFE else 0.0

                    a = CMD_SMOOTH_ALPHA
                    vx_s = (1-a)*vx_s + a*vx_cmd
                    vy_s = (1-a)*vy_s + a*vy_cmd
                    yaw_s = (1-a)*yaw_s + a*yaw_cmd

                    send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, yaw_s)

                    # stop detect -> approach
                    if mode == MODE_FRONT_FOLLOW:
                        t = time.time()
                        if last_bbox_for_stop is not None and last_bbox_time is not None:
                            px_prev = (last_bbox_for_stop[0] + last_bbox_for_stop[2]) * 0.5
                            bh_prev = max(1.0, (last_bbox_for_stop[3] - last_bbox_for_stop[1]))
                            if abs(px - px_prev) < STILL_CENTER_PX and abs(bh - bh_prev) < STILL_SIZE_PX:
                                stationary_t += (t - last_bbox_time)
                            else:
                                stationary_t = 0.0

                        last_bbox_for_stop = (x1,y1,x2,y2,cf)
                        last_bbox_time = t

                        if stationary_t >= STOP_SECONDS:
                            mode = MODE_FRONT_APPROACH
                            switch_stable_start = None
                            stationary_t = 0.0

                    # approach -> switch to DOWN when close+center
                    if mode == MODE_FRONT_APPROACH:
                        err_x_abs = abs((px - cx) / cx)
                        close_ok  = (bh >= CLOSE_TARGET_H_PX * CLOSE_H_TOL)
                        center_ok = (err_x_abs <= CENTER_X_TOL)

                        if center_ok and close_ok:
                            if switch_stable_start is None:
                                switch_stable_start = time.time()
                            elif (time.time() - switch_stable_start) >= SWITCH_STABLE_SEC:
                                stop_vehicle(vehicle, 0.10)
                                set_active_cam("DOWN")
                                mode = MODE_DOWN_FOLLOW
                                switch_stable_start = None
                                continue
                        else:
                            switch_stable_start = None

            if mode == MODE_DOWN_FOLLOW:
                if active_cam != "DOWN":
                    mode = MODE_FRONT_FOLLOW
                else:
                    if bbox_down is None or frame_down is None:
                        stop_vehicle(vehicle, 0.15)
                        time.sleep(1.0 / CONTROL_HZ)
                        continue

                    x1,y1,x2,y2,cf = bbox_down
                    px = (x1+x2)*0.5
                    py = (y1+y2)*0.5

                    fh, fw = frame_down.shape[:2]
                    cx, cy = fw*0.5, fh*0.5

                    err_x = (px - cx) / cx
                    err_y = (py - cy) / cy

                    if abs(err_x) < DOWN_DEADBAND: err_x = 0.0
                    if abs(err_y) < DOWN_DEADBAND: err_y = 0.0

                    vx_cmd = clamp(-Kp_down_xy * err_y, -MAX_VX, MAX_VX)
                    vy_cmd = clamp(+Kp_down_xy * err_x, -MAX_VY, MAX_VY)

                    a = CMD_SMOOTH_ALPHA
                    vx_s = (1-a)*vx_s + a*vx_cmd
                    vy_s = (1-a)*vy_s + a*vy_cmd

                    send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, 0.0)

            # ============================================================
            # DISPLAY
            # ============================================================
            if DISPLAY:
                disp = active_frame.copy()
                if mode == MODE_FRONT_FOLLOW:
                    extra = f"stop-detect:{stationary_t:.1f}/{STOP_SECONDS}s"
                elif mode == MODE_FRONT_APPROACH:
                    extra = "APPROACHING..."
                else:
                    extra = "FOLLOWING ON DOWN CAM"

                draw_overlay(disp, bbox_active, active_cam, mode, alt, vx_s, vy_s, yaw_s, extra)
                cv2.imshow("ACTIVE Camera (DualCam Follow)", disp)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('s'):
                    stop_vehicle(vehicle, 1.0)
                if k == ord('l'):
                    set_mode_safe(vehicle, "LAND")
                    break

            time.sleep(1.0 / CONTROL_HZ)

    finally:
        try:
            if vehicle:
                try:
                    stop_vehicle(vehicle, 1.0)
                except Exception:
                    pass
        finally:
            if cam_front:
                cam_front.close()
            if cam_down and cam_down is not cam_front:
                cam_down.close()
            if DISPLAY:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if vehicle:
                try:
                    vehicle.close()
                except Exception:
                    pass

if __name__ == "__main__":
    main()
