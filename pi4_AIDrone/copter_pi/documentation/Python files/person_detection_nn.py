#!/usr/bin/env python3
"""
SCRIPT 2: Person Follow (YOLO11) - 640x480
=========================================
- Takeoff to 5m
- Use YOLO11 to detect person (class 0)
- Track target (largest box)
- Control:
    yaw_rate = Kp_yaw * horizontal_error
    vx       = Kp_dist * (target_box_height - current_box_height)
  optional:
    vy       = Kp_strafe * horizontal_error

SITL note:
- The "camera" is your laptop/webcam or a video file; the drone in SITL will move based on it.
REAL note:
- Use onboard camera feed on the companion computer.
"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time, math
import cv2
from ultralytics import YOLO

# =========================
# MODE: SITL or REAL
# =========================
USE_SITL = False   # set False on real drone


if USE_SITL:
    CONNECTION_STRING = "tcp:127.0.0.1:5762"
    BAUD_RATE = None
else:
    CONNECTION_STRING = '/dev/ttyACM0'
    BAUD_RATE = 115200

# =========================
# CAMERA / MODEL
# =========================
CAM_INDEX = 0                 # webcam
FRAME_W, FRAME_H = 640, 480
MODEL_PATH = "./yolo11n_ncnn_model"     # or your TRT engine path if using
CONF_THRES = 0.35

# =========================
# FLIGHT TARGETS (SAFETY)
# =========================
TARGET_ALTITUDE = 3

SEND_HZ = 15                   # smooth control
TYPE_MASK_VEL_YAWRATE = 0x05C7 # ignore pos+accel+yaw, use vel + yaw_rate

MAX_VX = 1.0                   # m/s (limit!)
MAX_VY = 0.6                   # m/s
MAX_YAW_RATE = 25.0            # deg/s

# Distance proxy using box height (pixels)
# You must calibrate this:
# 1) Hover at desired distance, record person bbox height ~TARGET_H_PX
TARGET_H_PX = 180

# Gains (start small)
Kp_yaw = 18.0                  # deg/s per normalized error
Kp_dist = 0.010                # m/s per pixel error
Kp_strafe = 0.40               # m/s per normalized error (optional)

USE_STRAFE = True

# Tracking loss safety
LOST_TIMEOUT_S = 0.6

# If yaw direction is opposite on your setup, flip this
YAW_SIGN = 1

# =========================
# Helpers
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def connect_vehicle():
    if BAUD_RATE:
        v = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True, timeout=60)
    else:
        v = connect(CONNECTION_STRING, wait_ready=True, timeout=60)
    print("Connected ✅", v.version)
    return v

def wait_until_armable(v):
    while not v.is_armable:
        time.sleep(1)

def wait_for_gps(v):
    while getattr(v, "gps_0", None) and v.gps_0.fix_type < 3:
        time.sleep(1)

def arm_and_takeoff(v, alt):
    v.mode = VehicleMode("GUIDED")
    while v.mode.name != "GUIDED":
        time.sleep(0.2)

    wait_until_armable(v)
    if not USE_SITL:
        wait_for_gps(v)

    v.armed = True
    while not v.armed:
        time.sleep(0.2)

    v.simple_takeoff(alt)
    while True:
        a = v.location.global_relative_frame.alt
        if a >= 0.95 * alt:
            break
        time.sleep(1)
    time.sleep(2)

def send_body_vel_yawrate(v, vx, vy, vz, yaw_rate_deg_s):
    yaw_rate_rad_s = math.radians(yaw_rate_deg_s) * YAW_SIGN
    msg = v.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,   # body frame velocities :contentReference[oaicite:10]{index=10}
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate_rad_s
    )
    v.send_mavlink(msg)
    v.flush()

def stop(v, duration=1.0):
    period = 1.0 / SEND_HZ
    end = time.time() + duration
    while time.time() < end:
        send_body_vel_yawrate(v, 0, 0, 0, 0)
        time.sleep(period)

# =========================
# Main follow loop
# =========================
def main():
    vehicle = None
    cap = None
    try:
        # Camera
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        # Model
        model = YOLO(MODEL_PATH)

        # Vehicle
        vehicle = connect_vehicle()
        arm_and_takeoff(vehicle, TARGET_ALTITUDE)

        last_seen = 0.0
        vx_s, vy_s, yaw_s = 0.0, 0.0, 0.0  # smoothed commands
        alpha = 0.35  # smoothing factor

        period = 1.0 / SEND_HZ

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            cx = FRAME_W // 2
            cy = FRAME_H // 2

            # YOLO inference (detect persons only)
            results = model.predict(frame, conf=CONF_THRES, classes=[0], verbose=False)
            boxes = results[0].boxes

            target = None
            best_area = 0

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        target = (x1, y1, x2, y2)

            if target is None:
                # Lost target -> stop safely if timeout exceeded
                if time.time() - last_seen > LOST_TIMEOUT_S:
                    vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, 0.0
                    send_body_vel_yawrate(vehicle, vx_cmd, vy_cmd, 0.0, yaw_cmd)
                cv2.putText(frame, "NO PERSON (stopping if lost)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                last_seen = time.time()
                x1, y1, x2, y2 = target
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)
                h = int(y2 - y1)

                # Normalized horizontal error [-1..+1]
                err_x = (px - cx) / float(cx)

                # Yaw rate to center the person
                yaw_cmd = clamp(Kp_yaw * err_x, -MAX_YAW_RATE, MAX_YAW_RATE)

                # Forward/back using bbox height as distance proxy
                err_h = (TARGET_H_PX - h)  # if person too small -> move forward (+vx)
                vx_cmd = clamp(Kp_dist * err_h, -MAX_VX, MAX_VX)

                # Optional gentle strafe
                if USE_STRAFE:
                    vy_cmd = clamp(Kp_strafe * err_x, -MAX_VY, MAX_VY)
                else:
                    vy_cmd = 0.0

                # Smooth commands (reduces twitch)
                vx_s = (1 - alpha) * vx_s + alpha * vx_cmd
                vy_s = (1 - alpha) * vy_s + alpha * vy_cmd
                yaw_s = (1 - alpha) * yaw_s + alpha * yaw_cmd

                send_body_vel_yawrate(vehicle, vx_s, vy_s, 0.0, yaw_s)

                # Draw HUD
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.circle(frame, (px, py), 4, (255,0,0), -1)
                cv2.line(frame, (cx, 0), (cx, FRAME_H), (200,200,200), 1)
                cv2.putText(frame, f"err_x={err_x:+.2f}  h={h}px  vx={vx_s:+.2f} vy={vy_s:+.2f} yaw={yaw_s:+.1f}",
                            (10, FRAME_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Person Follow (YOLO11) 640x480", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                print("STOP")
                stop(vehicle, 1.0)
            if key == ord('l'):
                print("LAND")
                vehicle.mode = VehicleMode("LAND")
                break

            time.sleep(period)

        # Safe stop at end
        stop(vehicle, 1.0)

    except KeyboardInterrupt:
        print("\nEMERGENCY STOP -> LAND")
        if vehicle:
            vehicle.mode = VehicleMode("LAND")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if vehicle:
            vehicle.close()

if __name__ == "__main__":
    main()
