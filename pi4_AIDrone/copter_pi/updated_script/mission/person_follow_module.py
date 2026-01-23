#!/usr/bin/env python3
"""
person_follow_module.py
=======================
Refactored person-follow module (YOLO + body-frame velocity control).

Key refactor:
- Importable: PersonFollower(vehicle, config).run()
- Can reuse an existing Vehicle connection (single MAVLink connection for orchestrator)
- Standalone mode still supported for quick tests

Original behavior came from your uploaded person_follow.py. fileciteturn0file1
"""

from __future__ import annotations

import time
import math
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO
from dronekit import Vehicle, VehicleMode
from pymavlink import mavutil

from vehicle_utils import (
    VehicleConnectionConfig,
    connect_vehicle,
    ensure_guided_ready,
    arm,
    takeoff,
    close_vehicle,
)

LOG = logging.getLogger("person_follow_module")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PersonFollowConfig:
    cam_index: int = 0
    frame_w: int = 640
    frame_h: int = 480

    model_path: str = "yolo11n.pt"   # can be .pt or your TRT wrapper path if you load differently
    conf_thres: float = 0.35

    send_hz: int = 15
    type_mask_vel_yawrate: int = 0x05C7  # ignore pos+accel+yaw, use vel + yaw_rate

    max_vx: float = 1.0          # m/s
    max_vy: float = 0.6          # m/s
    max_yaw_rate: float = 25.0   # deg/s

    target_h_px: int = 180       # bbox height at desired distance (calibrate)

    kp_yaw: float = 18.0         # deg/s per normalized error
    kp_dist: float = 0.010       # m/s per pixel error
    kp_strafe: float = 0.40      # m/s per normalized error

    use_strafe: bool = True
    yaw_sign: int = 1

    lost_timeout_s: float = 0.6

    # UI
    display: bool = True
    window_name: str = "Person Follow (YOLO) 640x480"

    # Optional: if running standalone, you may want takeoff
    takeoff_alt_m: Optional[float] = None   # None => do not takeoff (assume already airborne)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PersonFollower:
    def __init__(self, vehicle: Vehicle, cfg: PersonFollowConfig):
        self.v = vehicle
        self.cfg = cfg
        self._stop = False

        # Lazy inits
        self._cap = None
        self._model = None

    def _ensure_resources(self):
        if self._cap is None:
            cap = cv2.VideoCapture(self.cfg.cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
            self._cap = cap

        if self._model is None:
            self._model = YOLO(self.cfg.model_path)

    def request_stop(self):
        self._stop = True

    def send_body_vel_yawrate(self, vx: float, vy: float, vz: float, yaw_rate_deg_s: float) -> None:
        yaw_rate_rad_s = math.radians(yaw_rate_deg_s) * int(self.cfg.yaw_sign)
        msg = self.v.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            self.cfg.type_mask_vel_yawrate,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0, float(yaw_rate_rad_s)
        )
        self.v.send_mavlink(msg)
        self.v.flush()

    def stop_motion(self, duration_s: float = 1.0) -> None:
        period = 1.0 / max(2, int(self.cfg.send_hz))
        end = time.time() + float(duration_s)
        while time.time() < end:
            self.send_body_vel_yawrate(0.0, 0.0, 0.0, 0.0)
            time.sleep(period)

    def run(self) -> None:
        """
        Main follow loop.
        Assumes vehicle is already in GUIDED and airborne unless cfg.takeoff_alt_m is set.
        """
        self._ensure_resources()

        # Smooth commands
        vx_s = vy_s = yaw_s = 0.0
        alpha = 0.35
        period = 1.0 / max(2, int(self.cfg.send_hz))

        last_seen = 0.0

        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                LOG.warning("Camera read failed")
                break

            frame = cv2.resize(frame, (self.cfg.frame_w, self.cfg.frame_h))
            cx = self.cfg.frame_w // 2

            results = self._model.predict(frame, conf=self.cfg.conf_thres, classes=[0], verbose=False)
            boxes = results[0].boxes

            target = None
            best_area = 0.0

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    area = float((x2 - x1) * (y2 - y1))
                    if area > best_area:
                        best_area = area
                        target = (float(x1), float(y1), float(x2), float(y2))

            if target is None:
                if time.time() - last_seen > self.cfg.lost_timeout_s:
                    self.send_body_vel_yawrate(0.0, 0.0, 0.0, 0.0)

                if self.cfg.display:
                    cv2.putText(frame, "NO PERSON (holding)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                last_seen = time.time()
                x1, y1, x2, y2 = target
                px = int((x1 + x2) / 2)
                h = int(y2 - y1)

                # Normalized horizontal error [-1..+1]
                err_x = (px - cx) / float(cx)

                yaw_cmd = clamp(self.cfg.kp_yaw * err_x, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

                # Box height proxy: if too small -> move forward
                err_h = (self.cfg.target_h_px - h)
                vx_cmd = clamp(self.cfg.kp_dist * err_h, -self.cfg.max_vx, self.cfg.max_vx)

                if self.cfg.use_strafe:
                    vy_cmd = clamp(self.cfg.kp_strafe * err_x, -self.cfg.max_vy, self.cfg.max_vy)
                else:
                    vy_cmd = 0.0

                vx_s = (1 - alpha) * vx_s + alpha * vx_cmd
                vy_s = (1 - alpha) * vy_s + alpha * vy_cmd
                yaw_s = (1 - alpha) * yaw_s + alpha * yaw_cmd

                self.send_body_vel_yawrate(vx_s, vy_s, 0.0, yaw_s)

                if self.cfg.display:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (px, int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
                    cv2.line(frame, (cx, 0), (cx, self.cfg.frame_h), (200, 200, 200), 1)
                    cv2.putText(frame, f"err_x={err_x:+.2f} h={h}px vx={vx_s:+.2f} vy={vy_s:+.2f} yaw={yaw_s:+.1f}",
                                (10, self.cfg.frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.cfg.display:
                cv2.imshow(self.cfg.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    LOG.info("STOP command")
                    self.stop_motion(1.0)
                if key == ord("l"):
                    LOG.info("LAND command")
                    self.v.mode = VehicleMode("LAND")
                    break

            time.sleep(period)

        # End loop -> stop
        self.stop_motion(1.0)

    def close(self):
        if self._cap is not None:
            self._cap.release()
        if self.cfg.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Standalone runner (optional)
# -----------------------------------------------------------------------------
def run_follow_standalone(sitl: bool, cfg: PersonFollowConfig):
    v = None
    follower = None
    try:
        v = connect_vehicle(VehicleConnectionConfig(sitl=sitl))
        ensure_guided_ready(v, sitl=sitl)

        # Optional takeoff when standalone
        if cfg.takeoff_alt_m is not None:
            arm(v)
            takeoff(v, float(cfg.takeoff_alt_m))
        else:
            # ensure armed/airborne externally; still safe to keep GUIDED
            pass

        follower = PersonFollower(v, cfg)
        follower.run()

    finally:
        if follower:
            follower.close()
        close_vehicle(v)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitl", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--takeoff", type=float, default=None, help="Takeoff altitude (m). If omitted, no takeoff.")
    ap.add_argument("--headless", action="store_true", help="Disable display window (Jetson headless).")
    ap.add_argument("--model", type=str, default="yolo11n.pt")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    args = _parse_args()
    sitl = True if args.sitl else False
    if args.real:
        sitl = False

    cfg = PersonFollowConfig(
        model_path=args.model,
        display=not args.headless,
        takeoff_alt_m=args.takeoff,
    )
    run_follow_standalone(sitl=sitl, cfg=cfg)


if __name__ == "__main__":
    main()
