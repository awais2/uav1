#!/usr/bin/env python3
"""
Script 3: GUIDED Takeoff, Hover & Land (GPS Required)
====================================================
Platform : Raspberry Pi
Autopilot: Pixhawk 6C
Firmware : ArduCopter 4.x
Mode     : GUIDED
"""

from dronekit import connect, VehicleMode
import time

# ============================================
# CONNECTION SETTINGS
# ============================================
CONNECTION_STRING = '/dev/ttyACM0'
BAUD_RATE = 115200
TARGET_ALTITUDE = 5.0   # meters
HOVER_TIME = 7          # seconds

# ============================================
# FUNCTIONS
# ============================================

def connect_vehicle():
    print("=" * 50)
    print("CONNECTING TO VEHICLE")
    print("=" * 50)

    vehicle = connect(
        CONNECTION_STRING,
        baud=BAUD_RATE,
        wait_ready=True,
        timeout=60
    )

    print("Connected successfully!")
    return vehicle


def print_status(vehicle):
    print("\n--- VEHICLE STATUS ---")
    print(f" Firmware : {vehicle.version}")
    print(f" Mode     : {vehicle.mode.name}")
    print(f" Armed    : {vehicle.armed}")
    print(f" Armable  : {vehicle.is_armable}")

    gps = vehicle.gps_0
    print(f" GPS      : Fix={gps.fix_type}, Sats={gps.satellites_visible}")

    if vehicle.battery:
        print(f" Battery  : {vehicle.battery.voltage}V")

    print("----------------------")


def wait_for_gps(vehicle):
    print("\nWaiting for GPS fix...")
    while vehicle.gps_0.fix_type < 3:
        print(
            f"  GPS Fix={vehicle.gps_0.fix_type}, "
            f"Sats={vehicle.gps_0.satellites_visible}"
        )
        time.sleep(1)

    print("GPS fix acquired ✅")


def wait_until_armable(vehicle):
    print("\nWaiting for vehicle to become armable...")
    while not vehicle.is_armable:
        print("  Vehicle not armable yet...")
        time.sleep(1)
    print("Vehicle is armable ✅")


def arm_and_takeoff(vehicle, target_altitude):
    print("\nSetting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)

    print("GUIDED mode active")

    wait_until_armable(vehicle)

    print("Arming motors...")
    vehicle.armed = True

    while not vehicle.armed:
        time.sleep(0.5)

    print("*** VEHICLE ARMED ***")
    time.sleep(5)

    print(f"\nTaking off to {target_altitude} meters...")
    vehicle.simple_takeoff(target_altitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")

        if alt >= target_altitude * 0.95:
            print("Target altitude reached ✅")
            break

        time.sleep(1)


def hover(vehicle, duration):
    print(f"\nHovering for {duration} seconds...")
    time.sleep(duration)


def land(vehicle):
    print("\nLanding...")
    vehicle.mode = VehicleMode("LAND")

    while vehicle.mode.name != "LAND":
        time.sleep(0.5)

    while vehicle.armed:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt:.2f} m")
        time.sleep(1)

    print("*** VEHICLE LANDED & DISARMED ***")


# ============================================
# MAIN
# ============================================

def main():
    vehicle = None

    try:
        vehicle = connect_vehicle()
        print_status(vehicle)

        wait_for_gps(vehicle)

        arm_and_takeoff(vehicle, TARGET_ALTITUDE)
        hover(vehicle, HOVER_TIME)
        #land(vehicle)

        print("\nMISSION COMPLETED SUCCESSFULLY ✅")

    except Exception as e:
        print(f"\nERROR: {e}")

    finally:
        if vehicle:
            print("\nClosing connection...")
            vehicle.close()
            print("Done!")


if __name__ == "__main__":
    print("\n" + "#" * 50)
    print("# SCRIPT 3: GUIDED TAKEOFF & LAND (GPS)")
    print("#" * 50 + "\n")
    main()
