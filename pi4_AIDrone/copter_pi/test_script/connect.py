#!/usr/bin/env python3
"""
Script 1: Vehicle Connection & Status Check
============================================
Platform : Raspberry Pi
Autopilot: Pixhawk 6C (USB Type-C)
Firmware : ArduPilot 4.x
"""

from dronekit import connect
import time

# ============================================
# CONNECTION SETTINGS (USB)
# ============================================
# Pixhawk 6C USB usually appears as:
# /dev/ttyACM0 or /dev/ttyACM1
CONNECTION_STRING = '/dev/ttyACM0'
BAUD_RATE = 115200

# ============================================
# FUNCTIONS
# ============================================

def connect_vehicle():
    print("=" * 50)
    print("CONNECTING TO VEHICLE")
    print("=" * 50)
    print(f"Connection: {CONNECTION_STRING}")

    vehicle = connect(
        CONNECTION_STRING,
        baud=BAUD_RATE,
        wait_ready=True,
        timeout=60
    )

    print("\nCONNECTED SUCCESSFULLY!")
    return vehicle


def print_status(vehicle):
    print("\n--- VEHICLE STATUS ---")
    print(f" Firmware : {vehicle.version}")
    print(f" Mode     : {vehicle.mode.name}")
    print(f" Armed    : {vehicle.armed}")
    print(f" Armable  : {vehicle.is_armable}")

    # Battery
    try:
        if vehicle.battery:
            print(f" Battery  : {vehicle.battery.voltage}V")
    except:
        pass

    # GPS
    try:
        gps = vehicle.gps_0
        print(f" GPS      : Fix={gps.fix_type}, Sats={gps.satellites_visible}")
    except:
        pass

    print("----------------------")


# ============================================
# MAIN
# ============================================

def main():
    vehicle = None
    try:
        vehicle = connect_vehicle()
        print_status(vehicle)

        print("\nConnection test successful ✅")

    except Exception as e:
        print(f"\nERROR: {e}")

    finally:
        if vehicle:
            print("\nClosing connection...")
            vehicle.close()
            print("Done!")


if __name__ == "__main__":
    print("\n" + "#" * 50)
    print("# SCRIPT 1: CONNECTION TEST")
    print("#" * 50 + "\n")
    main()
