#!/usr/bin/env python3
"""
Script 4: GUIDED Movement Control (REAL HARDWARE)
================================================
Pixhawk 6C + Raspberry Pi
ArduCopter 4.x
GPS REQUIRED

Arm -> Takeoff -> Move (F, B, L, R) -> Land
"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time

# ============================================
# SETTINGS (REAL HARDWARE)
# ============================================
CONNECTION_STRING = '/dev/ttyACM0'
BAUD_RATE = 115200

TARGET_ALTITUDE = 5     # meters
MOVE_SPEED = 2.0           # m/s
MOVE_DURATION = 3          # seconds per direction

# ============================================
# CONNECTION
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
    print(f" Firmware: {vehicle.version}")
    print(f" Mode    : {vehicle.mode.name}")

    return vehicle

# ============================================
# SAFETY CHECKS
# ============================================

def wait_for_gps(vehicle):
    print("\nWaiting for GPS fix...")
    while vehicle.gps_0.fix_type < 3:
        print(
            f" GPS Fix={vehicle.gps_0.fix_type}, "
            f"Sats={vehicle.gps_0.satellites_visible}"
        )
        time.sleep(1)
    print("GPS fix acquired ✅")


def wait_until_armable(vehicle):
    print("\nWaiting until vehicle is armable...")
    while not vehicle.is_armable:
        print(" Vehicle not armable yet...")
        time.sleep(1)
    print("Vehicle is armable ✅")

# ============================================
# ARM & TAKEOFF
# ============================================

def arm_and_takeoff(vehicle, altitude):
    print("\nSetting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.5)
    print("GUIDED mode active")

    wait_until_armable(vehicle)

    print("\nArming motors...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.5)
    print("Vehicle ARMED")

    time.sleep(3)

    print(f"\nTaking off to {altitude} meters...")
    vehicle.simple_takeoff(altitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")
        if alt >= altitude * 0.95:
            print("Target altitude reached ✅")
            break
        time.sleep(1)

# ============================================
# VELOCITY CONTROL (BODY FRAME)
# ============================================

def send_velocity(vehicle, vx, vy, vz, duration):
    """
    vx: Forward(+)/Backward(-)
    vy: Right(+)/Left(-)
    vz: Down(+)/Up(-)
    """
    end_time = time.time() + duration

    while time.time() < end_time:
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.1)


def stop(vehicle):
    send_velocity(vehicle, 0, 0, 0, 1)


def move_forward(vehicle):
    print("\n>>> FORWARD")
    send_velocity(vehicle, MOVE_SPEED, 0, 0, MOVE_DURATION)
    stop(vehicle)


def move_backward(vehicle):
    print("\n>>> BACKWARD")
    send_velocity(vehicle, -MOVE_SPEED, 0, 0, MOVE_DURATION)
    stop(vehicle)


def move_left(vehicle):
    print("\n>>> LEFT")
    send_velocity(vehicle, 0, -MOVE_SPEED, 0, MOVE_DURATION)
    stop(vehicle)


def move_right(vehicle):
    print("\n>>> RIGHT")
    send_velocity(vehicle, 0, MOVE_SPEED, 0, MOVE_DURATION)
    stop(vehicle)

# ============================================
# LAND
# ============================================

def land(vehicle):
    print("\nLanding...")
    vehicle.mode = VehicleMode("LAND")

    while vehicle.armed:
        print(f" Altitude: {vehicle.location.global_relative_frame.alt:.2f} m")
        time.sleep(1)

    print("Landed & disarmed ✅")

# ============================================
# MAIN
# ============================================

def main():
    vehicle = None

    try:
        vehicle = connect_vehicle()

        wait_for_gps(vehicle)

        arm_and_takeoff(vehicle, TARGET_ALTITUDE)

        print("\nStarting movement sequence")

        
        move_forward(vehicle)
        time.sleep(2)

        move_backward(vehicle)
        time.sleep(2)

        #move_left(vehicle)
        #time.sleep(2)

        #move_right(vehicle)
        #time.sleep(2)
    

        print("\nMovement complete")

        land(vehicle)

        print("\nMISSION COMPLETED SUCCESSFULLY ✅")

    except KeyboardInterrupt:
        print("\nEMERGENCY STOP!")
        if vehicle:
            vehicle.mode = VehicleMode("LAND")

    except Exception as e:
        print(f"\nERROR: {e}")
        if vehicle:
            vehicle.mode = VehicleMode("LAND")

    finally:
        if vehicle:
            print("\nClosing connection...")
            vehicle.close()

# ============================================

if __name__ == "__main__":
    print("\n" + "#" * 50)
    print("# SCRIPT 4: GUIDED MOVEMENT (REAL DRONE)")
    print("#" * 50)
    main()
