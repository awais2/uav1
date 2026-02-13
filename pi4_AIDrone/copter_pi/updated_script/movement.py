#!/usr/bin/env python3
"""
SCRIPT 1: Move + Rotate 360 (Safety Training)
=============================================
Forward, Backward, Left, Right, then smooth rotate 360 degrees.

Uses SET_POSITION_TARGET_LOCAL_NED with MAV_FRAME_BODY_NED:
- vx/vy/vz are in BODY frame (forward/right/down)
- yaw_rate rotates smoothly while holding position (vx=vy=0)

ArduPilot notes:
- BODY frame velocities are relative to current heading. :contentReference[oaicite:3]{index=3}
- Velocity commands should be resent continuously; vehicle stops if commands stop. :contentReference[oaicite:4]{index=4}
"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time, math

# =========================
# MODE: SITL or REAL
# =========================
USE_SITL = True   # set False on real drone

if USE_SITL:
    CONNECTION_STRING = "tcp:127.0.0.1:5762"
    BAUD_RATE = None
else:
    CONNECTION_STRING = "/dev/serial/by-id/USB_SERIAL_HERE"
    BAUD_RATE = 115200

# =========================
# SAFETY SETTINGS
# =========================
TARGET_ALTITUDE = 5.0     # meters (safe training height)
MOVE_SPEED = 1.0          # m/s (start low)
MOVE_DURATION = 3.0       # seconds each direction

YAW_RATE_DEG_S = 30.0     # deg/s (smooth)
ROTATE_DEG = 360.0

SEND_HZ = 10              # smooth control (>= 2Hz; 10Hz feels good) :contentReference[oaicite:5]{index=5}

# TYPE_MASK: ignore position + accel + yaw, USE velocity + yaw_rate
# Based on bit mapping (pos/vel/accel/yaw/yaw_rate) in ArduPilot docs. :contentReference[oaicite:6]{index=6}
TYPE_MASK_VEL_YAWRATE = 0x05C7  # 1479

# If yaw direction is opposite on your setup, flip this to -1
YAW_SIGN = 1

# =========================
# Helpers
# =========================
def connect_vehicle():
    print("=" * 60)
    print("CONNECTING")
    print("=" * 60)
    if BAUD_RATE:
        v = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True, timeout=60)
    else:
        v = connect(CONNECTION_STRING, wait_ready=True, timeout=60)

    print("Connected ✅")
    print(f"Firmware: {v.version}")
    print(f"Mode    : {v.mode.name}")
    return v

def wait_until_armable(v):
    while not v.is_armable:
        print("Waiting armable...")
        time.sleep(1)

def wait_for_gps(v):
    # On SITL this is instant; on real it matters.
    while getattr(v, "gps_0", None) and v.gps_0.fix_type < 3:
        print(f"GPS fix={v.gps_0.fix_type} sats={v.gps_0.satellites_visible}")
        time.sleep(1)

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

def send_body_vel_yawrate(v, vx, vy, vz, yaw_rate_deg_s):
    yaw_rate_rad_s = math.radians(yaw_rate_deg_s) * YAW_SIGN
    msg = v.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,    # body frame (forward/right/down) :contentReference[oaicite:7]{index=7}
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate_rad_s
    )
    v.send_mavlink(msg)
    v.flush()

def hold_for(v, seconds, vx=0, vy=0, vz=0, yaw_rate=0):
    period = 1.0 / SEND_HZ
    end = time.time() + seconds
    while time.time() < end:
        send_body_vel_yawrate(v, vx, vy, vz, yaw_rate)
        time.sleep(period)

def stop(v, seconds=1.0):
    hold_for(v, seconds, 0, 0, 0, 0)

def land(v):
    print("LAND...")
    v.mode = VehicleMode("LAND")
    while v.armed:
        print(f"Alt: {v.location.global_relative_frame.alt:.2f}m")
        time.sleep(1)
    print("Disarmed ✅")

# =========================
# Main sequence
# =========================
def main():
    v = None
    try:
        v = connect_vehicle()
        arm_and_takeoff(v, TARGET_ALTITUDE)

        print("\nForward")
        hold_for(v, MOVE_DURATION, vx=+MOVE_SPEED); stop(v)

        print("Backward")
        hold_for(v, MOVE_DURATION, vx=-MOVE_SPEED); stop(v)

        print("Left")
        hold_for(v, MOVE_DURATION, vy=-MOVE_SPEED); stop(v)

        print("Right")
        hold_for(v, MOVE_DURATION, vy=+MOVE_SPEED); stop(v)

        # Rotate in place (vx=vy=0) using yaw_rate
        rotate_time = abs(ROTATE_DEG / YAW_RATE_DEG_S)
        print(f"\nRotate {ROTATE_DEG}° at {YAW_RATE_DEG_S}°/s (≈{rotate_time:.1f}s)")
        hold_for(v, rotate_time, vx=0, vy=0, vz=0, yaw_rate=YAW_RATE_DEG_S)
        stop(v, 2.0)

        print("Landing...")
        land(v)
        

        print("\nDONE ✅")
        # land(v)  # uncomment when you want auto-land

    except KeyboardInterrupt:
        print("\nEMERGENCY: LAND")
        if v:
            v.mode = VehicleMode("LAND")
    finally:
        if v:
            v.close()

if __name__ == "__main__":
    main()
