#!/usr/bin/env python3
"""
Script 2: Arm & Disarm (Indoor / NO GPS)
=======================================
Firmware : ArduCopter 4.5.5
Mode     : ALT_HOLD
"""

from dronekit import connect
import time

# ============================================
# CONNECTION SETTINGS
# ============================================
CONNECTION_STRING = '/dev/ttyACM0'
BAUD_RATE = 115200

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

    try:
        gps = vehicle.gps_0
        print(f" GPS      : Fix={gps.fix_type}, Sats={gps.satellites_visible}")
    except:
        pass

    try:
        if vehicle.battery:
            print(f" Battery  : {vehicle.battery.voltage}V")
    except:
        pass

    print("----------------------")


def wait_until_armable(vehicle):
    print("\nWaiting for vehicle to become armable...")
    while not vehicle.is_armable:
        print("  → Vehicle not armable yet...")
        time.sleep(1)
    print("Vehicle is now armable ✅")


def arm_vehicle(vehicle):
    print("\nSetting ALT_HOLD mode (NO GPS)...")
    vehicle.mode = 'ALT_HOLD'

    while vehicle.mode.name != 'ALT_HOLD':
        time.sleep(0.5)

    print("ALT_HOLD mode active")

    wait_until_armable(vehicle)

    print("Arming motors...")
    vehicle.armed = True

    while not vehicle.armed:
        time.sleep(0.2)

    print("\n*** VEHICLE ARMED (NO GPS) ***")


def disarm_vehicle(vehicle):
    print("\nDisarming motors...")
    vehicle.armed = False

    while vehicle.armed:
        time.sleep(0.2)

    print("*** VEHICLE DISARMED ***")


# ============================================
# MAIN
# ============================================

def main():
    vehicle = None

    try:
        vehicle = connect_vehicle()

        print_status(vehicle)

        arm_vehicle(vehicle)

        print("\nHolding armed state for 10 seconds...")
        time.sleep(10)

        disarm_vehicle(vehicle)

        print("\nARM/DISARM TEST COMPLETED ✅")

    except Exception as e:
        print(f"\nERROR: {e}")

    finally:
        if vehicle:
            print("\nClosing connection...")
            vehicle.close()
            print("Done!")


if __name__ == "__main__":
    print("\n" + "#" * 50)
    print("# SCRIPT 2: ARM & DISARM (ALT_HOLD)")
    print("#" * 50 + "\n")
    main()
