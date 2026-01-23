#!/usr/bin/env python3
"""
waypoint_module.py
==================
Refactored waypoint navigation module (DroneKit / ArduPilot style).

- Importable: WaypointNavigator(vehicle)
- Still runnable as a script for quick testing in SITL/real

Original behavior came from your uploaded waypoint script. fileciteturn0file0
"""

from __future__ import annotations

import time
import logging
import argparse
from dataclasses import dataclass
from typing import Iterable, Optional, List

from dronekit import Vehicle, LocationGlobalRelative

from vehicle_utils import (
    VehicleConnectionConfig,
    connect_vehicle,
    ensure_guided_ready,
    arm,
    takeoff,
    land,
    close_vehicle,
    distance_to_m,
)

LOG = logging.getLogger("waypoint_module")


@dataclass(frozen=True)
class Waypoint:
    lat: float
    lon: float
    alt: float  # meters AGL (global-relative)


@dataclass(frozen=True)
class ArrivalCriteria:
    radius_m: float = 3.0          # horizontal radius
    groundspeed_max: float = 0.5   # m/s
    dwell_s: float = 4.0           # must stay inside radius for this long
    alt_tol_m: float = 5.0         # acceptable altitude error
    timeout_s: float = 300.0       # mission timeout per waypoint
    poll_s: float = 0.2            # loop cadence


class WaypointNavigator:
    """
    Navigation helper that uses vehicle.simple_goto and robust arrival checks.
    Designed to be controlled by mission_manager.py (single MAVLink connection).
    """

    def __init__(self, vehicle: Vehicle):
        self.v = vehicle

    def goto(self, wp: Waypoint, groundspeed: Optional[float] = None) -> None:
        """Command a move to a waypoint."""
        if groundspeed is not None:
            self.v.groundspeed = float(groundspeed)

        target = LocationGlobalRelative(wp.lat, wp.lon, wp.alt)
        LOG.info("GOTO: lat=%.7f lon=%.7f alt=%.1f", wp.lat, wp.lon, wp.alt)
        self.v.simple_goto(target)

    def wait_arrival(self, wp: Waypoint, c: ArrivalCriteria) -> None:
        """
        Arrived when:
          - distance <= radius_m
          - groundspeed <= groundspeed_max
          - altitude within alt_tol_m
          - all held for dwell_s seconds
        """
        t0 = time.time()
        inside_since = None

        while True:
            dist = distance_to_m(self.v, wp.lat, wp.lon)
            alt = self.v.location.global_relative_frame.alt
            gs = float(getattr(self.v, "groundspeed", 0.0))

            alt_ok = abs(alt - wp.alt) <= c.alt_tol_m
            inside = (dist <= c.radius_m) and (gs <= c.groundspeed_max) and alt_ok

            if inside:
                if inside_since is None:
                    inside_since = time.time()
                elif (time.time() - inside_since) >= c.dwell_s:
                    LOG.info("ARRIVED ✅ dist=%.2fm gs=%.2fm/s alt=%.1f", dist, gs, alt)
                    return
            else:
                inside_since = None

            if time.time() - t0 > c.timeout_s:
                raise TimeoutError(
                    f"Arrival timeout: dist={dist:.1f}m gs={gs:.2f} alt={alt:.1f} (target alt={wp.alt:.1f})"
                )

            LOG.info("...dist=%.1fm gs=%.2f alt=%.1f", dist, gs, alt)
            time.sleep(c.poll_s)

    def descend_to(self, target_alt_m: float, tol_m: float = 1.0, timeout_s: float = 120.0) -> None:
        """Descend (or climb) to target altitude while holding lat/lon."""
        loc = self.v.location.global_relative_frame
        target = LocationGlobalRelative(loc.lat, loc.lon, float(target_alt_m))
        self.v.simple_goto(target)
        t0 = time.time()
        while True:
            alt = self.v.location.global_relative_frame.alt
            if abs(alt - target_alt_m) <= tol_m:
                LOG.info("Reached altitude %.1fm", alt)
                return
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Altitude change timeout: alt={alt:.1f}, target={target_alt_m:.1f}")
            time.sleep(1.0)


# -----------------------------------------------------------------------------
# Standalone test runner (SITL or real)
# -----------------------------------------------------------------------------
def run_waypoints_standalone(
    sitl: bool,
    waypoints: List[Waypoint],
    takeoff_alt_m: float = 30.0,
    groundspeed: float = 1.0,
) -> None:
    v = None
    try:
        v = connect_vehicle(VehicleConnectionConfig(sitl=sitl))
        ensure_guided_ready(v, sitl=sitl)
        arm(v)
        takeoff(v, takeoff_alt_m)

        nav = WaypointNavigator(v)
        criteria = ArrivalCriteria(radius_m=3.0, groundspeed_max=0.7, dwell_s=4.0, alt_tol_m=6.0, timeout_s=300.0)

        for wp in waypoints:
            nav.goto(wp, groundspeed=groundspeed)
            nav.wait_arrival(wp, criteria)

        # Optional: descend to 5m then land (matches your old flow)
        nav.descend_to(5.0)
        land(v)

    finally:
        close_vehicle(v)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitl", action="store_true", help="Use SITL tcp connection")
    ap.add_argument("--real", action="store_true", help="Use real hardware serial connection")
    ap.add_argument("--lat", type=float, default=33.7496616)
    ap.add_argument("--lon", type=float, default=73.1676921)
    ap.add_argument("--alt", type=float, default=30.0)
    ap.add_argument("--takeoff_alt", type=float, default=30.0)
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    args = _parse_args()
    sitl = True if args.sitl else False
    if args.real:
        sitl = False

    wps = [Waypoint(args.lat, args.lon, args.alt)]
    run_waypoints_standalone(sitl=sitl, waypoints=wps, takeoff_alt_m=args.takeoff_alt)


if __name__ == "__main__":
    main()
