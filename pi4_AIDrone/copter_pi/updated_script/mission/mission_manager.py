#!/usr/bin/env python3
"""
mission_manager.py
==================
Production-style orchestrator for:
1) Takeoff (optional)
2) Navigate to a GPS target (lat/lon/alt)
3) Detect arrival robustly (radius + speed + dwell + altitude tolerance)
4) Start person-follow behavior automatically

Design principles:
- Single MAVLink/DroneKit connection (prevents client conflicts)
- Clear state machine + timeouts
- Fail-safe behavior on errors/timeouts
- Works in SITL first (SITL=True), then switch to real hardware (SITL=False)

Dependencies:
- dronekit
- pymavlink
- opencv-python (for follow UI)
- ultralytics (YOLO) for person_follow_module

Usage examples:
  # SITL
  python3 mission_manager.py --sitl --lat 33.7496616 --lon 73.1676921 --alt 30 --takeoff-alt 30 --model yolo11n.pt

  # Real hardware (edit your serial port OR pass --conn)
  python3 mission_manager.py --real --conn /dev/ttyACM0 --lat 33.7496616 --lon 73.1676921 --alt 25 --takeoff-alt 25

Notes:
- This orchestrator assumes ArduPilot-style GUIDED control (DroneKit).
- For PX4 OFFBOARD, the control stack differs (MAVSDK setpoint streaming).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dronekit import VehicleMode

from vehicle_utils import (
    VehicleConnectionConfig,
    connect_vehicle,
    close_vehicle,
    ensure_guided_ready,
    arm,
    takeoff,
    descend_to_5m,
    land,
    set_mode,
)

from waypoint_module import Waypoint, ArrivalCriteria, WaypointNavigator
from person_follow_module import PersonFollowConfig, PersonFollower


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("mission_manager")


# -----------------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------------
class State(str, Enum):
    INIT = "INIT"
    CONNECT = "CONNECT"
    PREARM = "PREARM"
    TAKEOFF = "TAKEOFF"
    GOTO_TARGET = "GOTO_TARGET"
    ARRIVAL_CHECK = "ARRIVAL_CHECK"
    DESCEND_to_5m = "DESCEND_to_5m"
    START_FOLLOW = "START_FOLLOW"
    FOLLOW_RUNNING = "FOLLOW_RUNNING"
    END = "END"
    FAILSAFE = "FAILSAFE"


# -----------------------------------------------------------------------------
# Mission config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MissionConfig:
    sitl: bool
    conn: Optional[str]
    baud: int

    target_lat: float
    target_lon: float
    target_alt_m: float

    do_takeoff: bool
    takeoff_alt_m: float
    groundspeed: float

    arrival: ArrivalCriteria

    follow_enabled: bool
    follow_mode: str  # inline only (recommended)
    follow_seconds: Optional[float]  # None => run until user stops (UI q) or crash
    follow_cfg: PersonFollowConfig

    failsafe: str  # rtl|land|loiter|brake


def _setup_logging(logfile: Optional[str], level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _failsafe(vehicle, action: str) -> None:
    """Best-effort failsafe action."""
    action = (action or "rtl").lower()
    try:
        if action == "land":
            LOG.warning("FAILSAFE: LAND")
            land(vehicle)
        elif action == "loiter":
            LOG.warning("FAILSAFE: LOITER")
            set_mode(vehicle, "LOITER")
        elif action == "brake":
            # ArduPilot copter supports BRAKE; if not available, LOITER is okay.
            LOG.warning("FAILSAFE: BRAKE")
            try:
                set_mode(vehicle, "BRAKE")
            except Exception:
                set_mode(vehicle, "LOITER")
        else:
            LOG.warning("FAILSAFE: RTL")
            set_mode(vehicle, "RTL")
    except Exception:
        LOG.exception("Failsafe action failed; last resort: try LAND")
        try:
            set_mode(vehicle, "LAND")
        except Exception:
            pass


def run_mission(cfg: MissionConfig) -> int:
    state: State = State.INIT
    vehicle = None
    follower: Optional[PersonFollower] = None

    try:
        state = State.CONNECT

        vcfg = VehicleConnectionConfig(
            sitl=cfg.sitl,
            sitl_connection_string="tcp:127.0.0.1:5762",
            real_connection_string=cfg.conn or "/dev/ttyACM0",
            baud=cfg.baud,
            timeout=60,
            wait_ready=True,
        )

        vehicle = connect_vehicle(vcfg)

        # ---------------------------------------------------------------------
        state = State.PREARM
        ensure_guided_ready(vehicle, sitl=cfg.sitl)

        if cfg.do_takeoff:
            LOG.info("Arming...")
            arm(vehicle)
            LOG.info("Takeoff to %.1fm", cfg.takeoff_alt_m)
            takeoff(vehicle, cfg.takeoff_alt_m)
        else:
            LOG.info("Skipping takeoff (do_takeoff=False). Assuming already airborne & GUIDED.")

        # ---------------------------------------------------------------------
        state = State.GOTO_TARGET
        nav = WaypointNavigator(vehicle)
        wp = Waypoint(cfg.target_lat, cfg.target_lon, cfg.target_alt_m)
        nav.goto(wp, groundspeed=cfg.groundspeed)

        # ---------------------------------------------------------------------
        state = State.ARRIVAL_CHECK
        nav.wait_arrival(wp, cfg.arrival)

        # ---------------------------------------------------------------------
        state = State.DESCEND_to_5m
        descend_to_5m(vehicle)
        #----------------------------------------------------------------------
        if not cfg.follow_enabled:
            LOG.info("Follow disabled; mission completed at target.")
            state = State.END
            return 0

        state = State.START_FOLLOW
        # Ensure GUIDED for follow control
        try:
            vehicle.mode = VehicleMode("GUIDED")
        except Exception:
            pass

        follower = PersonFollower(vehicle, cfg.follow_cfg)

        # ---------------------------------------------------------------------
        state = State.FOLLOW_RUNNING
        LOG.info("Starting person follow (%s)...", cfg.follow_mode)

        if cfg.follow_mode != "inline":
            raise ValueError("Only follow_mode='inline' is supported in this professional single-connection setup.")

        if cfg.follow_seconds is None:
            follower.run()  # run until 'q' in UI or crash / stop requested
        else:
            # run with a time budget (for headless missions)
            t_end = time.time() + float(cfg.follow_seconds)
            while time.time() < t_end:
                follower.run()
                break  # follower.run() is blocking; if it exits, we stop early

        state = State.END
        LOG.info("Mission complete ✅")
        return 0

    except KeyboardInterrupt:
        LOG.warning("KeyboardInterrupt -> stopping")
        if follower:
            try:
                follower.request_stop()
            except Exception:
                pass
        if vehicle:
            _failsafe(vehicle, cfg.failsafe)
        return 130

    except Exception as e:
        LOG.exception("Mission failed in state=%s: %s", state, e)
        if vehicle:
            state = State.FAILSAFE
            _failsafe(vehicle, cfg.failsafe)
        return 1

    finally:
        # Stop follower motion and close UI/camera
        if follower:
            try:
                follower.request_stop()
                follower.close()
            except Exception:
                pass
        if vehicle:
            close_vehicle(vehicle)


def _parse_args():
    ap = argparse.ArgumentParser(description="Waypoint -> Arrival -> Person Follow Orchestrator")

    # Mode
    ap.add_argument("--sitl", action="store_true", help="Use SITL connection (tcp:127.0.0.1:5762)")
    ap.add_argument("--real", action="store_true", help="Use real hardware serial connection")

    # Connection
    ap.add_argument("--conn", type=str, default=None, help="Serial device for real hardware (e.g. /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200)

    # Target waypoint
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--alt", type=float, default=30.0, help="Target altitude (m AGL, global-relative)")

    # Takeoff
    ap.add_argument("--takeoff-alt", type=float, default=30.0)
    ap.add_argument("--no-takeoff", action="store_true", help="Skip arming + takeoff (assume already airborne)")

    # Navigation tuning
    ap.add_argument("--groundspeed", type=float, default=1.0)
    ap.add_argument("--arr-radius", type=float, default=3.0)
    ap.add_argument("--arr-gs", type=float, default=0.7)
    ap.add_argument("--arr-dwell", type=float, default=4.0)
    ap.add_argument("--arr-alt-tol", type=float, default=6.0)
    ap.add_argument("--arr-timeout", type=float, default=300.0)

    # Follow
    ap.add_argument("--no-follow", action="store_true", help="Do not start person follow after arrival")
    ap.add_argument("--follow-seconds", type=float, default=None, help="Run follow for N seconds then exit")
    ap.add_argument("--follow-mode", type=str, default="inline", choices=["inline"], help="inline is recommended")

    # Follow config
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--model", type=str, default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--headless", action="store_true", help="Disable OpenCV window (Jetson/RPi headless)")

    # Failsafe
    ap.add_argument("--failsafe", type=str, default="rtl", choices=["rtl", "land", "loiter", "brake"])

    # Logging
    ap.add_argument("--log", type=str, default="mission_manager.log")
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return ap.parse_args()


def main():
    args = _parse_args()
    _setup_logging(args.log, args.log_level)

    # Decide SITL vs real
    sitl = True if args.sitl else False
    if args.real:
        sitl = False

    arrival = ArrivalCriteria(
        radius_m=float(args.arr_radius),
        groundspeed_max=float(args.arr_gs),
        dwell_s=float(args.arr_dwell),
        alt_tol_m=float(args.arr_alt_tol),
        timeout_s=float(args.arr_timeout),
        poll_s=0.2,
    )

    follow_cfg = PersonFollowConfig(
        cam_index=int(args.cam),
        frame_w=int(args.w),
        frame_h=int(args.h),
        model_path=args.model,
        conf_thres=float(args.conf),
        display=not bool(args.headless),
        takeoff_alt_m=None,  # orchestrator already handles takeoff
    )

    cfg = MissionConfig(
        sitl=sitl,
        conn=args.conn,
        baud=int(args.baud),
        target_lat=float(args.lat),
        target_lon=float(args.lon),
        target_alt_m=float(args.alt),
        do_takeoff=not bool(args.no_takeoff),
        takeoff_alt_m=float(args.takeoff_alt),
        groundspeed=float(args.groundspeed),
        arrival=arrival,
        follow_enabled=not bool(args.no_follow),
        follow_mode=args.follow_mode,
        follow_seconds=args.follow_seconds,
        follow_cfg=follow_cfg,
        failsafe=args.failsafe,
    )

    LOG.info("Starting mission (sitl=%s) -> target=(%.7f, %.7f, %.1fm)", cfg.sitl, cfg.target_lat, cfg.target_lon, cfg.target_alt_m)
    rc = run_mission(cfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()
