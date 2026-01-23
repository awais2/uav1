#!/usr/bin/env python3
"""
vehicle_utils.py
================
Shared DroneKit helpers for both waypoint navigation and person following.

Design goals:
- One place for connection logic (SITL vs Real)
- Safe, time-bounded waits (armable, GPS fix, mode changes)
- Simple geo distance helpers (meters)
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from dronekit import connect, Vehicle, VehicleMode, LocationGlobalRelative


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger("vehicle_utils")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VehicleConnectionConfig:
    sitl: bool = True
    sitl_connection_string: str = "tcp:127.0.0.1:5762"
    real_connection_string: str = "/dev/serial/by-id/USB_SERIAL_HERE"  # change on hardware
    baud: int = 115200
    timeout: int = 60
    wait_ready: bool = True


# -----------------------------------------------------------------------------
# Geo helpers
# -----------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates in meters."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def distance_to_m(vehicle: Vehicle, lat: float, lon: float) -> float:
    """Distance from vehicle current GLOBAL_RELATIVE lat/lon to target in meters."""
    loc = vehicle.location.global_relative_frame
    return haversine_m(loc.lat, loc.lon, lat, lon)


# -----------------------------------------------------------------------------
# Connection & state helpers
# -----------------------------------------------------------------------------
def connect_vehicle(cfg: VehicleConnectionConfig) -> Vehicle:
    """Connect to Pixhawk via DroneKit."""
    conn = cfg.sitl_connection_string if cfg.sitl else cfg.real_connection_string
    LOG.info("Connecting to vehicle: %s (sitl=%s)", conn, cfg.sitl)
    if cfg.sitl:
        v = connect(conn, wait_ready=cfg.wait_ready, timeout=cfg.timeout)
    else:
        v = connect(conn, baud=cfg.baud, wait_ready=cfg.wait_ready, timeout=cfg.timeout)
    LOG.info("Connected ✅  Firmware=%s  Mode=%s", getattr(v, "version", "?"), v.mode.name)
    return v


def set_mode(vehicle: Vehicle, mode_name: str, timeout_s: float = 10.0) -> None:
    """Set vehicle mode and wait for confirmation."""
    vehicle.mode = VehicleMode(mode_name)
    t0 = time.time()
    while vehicle.mode.name != mode_name:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Mode change timeout: wanted {mode_name}, got {vehicle.mode.name}")
        time.sleep(0.2)


def wait_until_armable(vehicle: Vehicle, timeout_s: float = 60.0) -> None:
    """Wait until vehicle.is_armable becomes True."""
    t0 = time.time()
    while not vehicle.is_armable:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Vehicle not armable within timeout")
        time.sleep(1.0)


def wait_for_gps_fix(vehicle: Vehicle, min_fix_type: int = 3, timeout_s: float = 120.0) -> None:
    """
    Wait for GPS fix_type >= min_fix_type.
    NOTE: Many SITL setups report GPS immediately; on real hardware this is important.
    """
    t0 = time.time()
    while getattr(vehicle, "gps_0", None) and vehicle.gps_0.fix_type < min_fix_type:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"GPS fix not reached within timeout (need fix_type>={min_fix_type})")
        LOG.info("Waiting GPS... fix=%s sats=%s", vehicle.gps_0.fix_type, vehicle.gps_0.satellites_visible)
        time.sleep(1.0)


def arm(vehicle: Vehicle, timeout_s: float = 15.0) -> None:
    """Arm vehicle and wait."""
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Arm timeout")
        time.sleep(0.2)


def takeoff(vehicle: Vehicle, target_alt_m: float, timeout_s: float = 60.0) -> None:
    """Simple takeoff to altitude (meters AGL). Requires GUIDED mode and arming."""
    vehicle.simple_takeoff(target_alt_m)
    t0 = time.time()
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= 0.95 * target_alt_m:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Takeoff timeout: altitude={alt:.1f}m target={target_alt_m:.1f}m")
        time.sleep(1.0)
    time.sleep(1.0)


def ensure_guided_ready(vehicle: Vehicle, sitl: bool) -> None:
    """
    Prepare for companion control:
    - switch to GUIDED
    - wait armable
    - wait GPS fix on real
    """
    set_mode(vehicle, "GUIDED")
    wait_until_armable(vehicle)
    if not sitl:
        wait_for_gps_fix(vehicle)

def descend_to_5m(vehicle):
    """Descend the drone to 5 meters after reaching the target waypoint."""
    print(f"Descending to 5 meters...")
    vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, 5.0))
    while abs(vehicle.location.global_relative_frame.alt - 5.0) > 0.5:
        print(f"Current altitude: {vehicle.location.global_relative_frame.alt}m")
        time.sleep(1)
    print("Reached 5 meters.")
    
def land(vehicle: Vehicle, timeout_s: float = 180.0) -> None:
    """Land and wait for disarm (best-effort)."""
    set_mode(vehicle, "LAND")
    t0 = time.time()
    while vehicle.armed:
        if time.time() - t0 > timeout_s:
            LOG.warning("Land timeout - still armed; check vehicle state.")
            break
        time.sleep(1.0)


def close_vehicle(vehicle: Optional[Vehicle]) -> None:
    if vehicle is None:
        return
    try:
        vehicle.close()
    except Exception:
        LOG.exception("Error closing vehicle")
