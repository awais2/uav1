import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

# =========================
# CONFIGURATION SETTINGS
# =========================
USE_SITL = False   # Set to False for real drone, True for SITL
TARGET_ALTITUDE = 20.0  # meters (safe altitude for training)
MOVE_SPEED = 1.0       # m/s (start low)
SEND_HZ = 10           # smooth control (>= 2Hz; 10Hz feels good)
GPS_TARGETS = [
    (33.7496616, 73.1676921, 30)  # Example target coordinates
]

# =========================
# CONNECTION SETTINGS
# =========================
if USE_SITL:
    CONNECTION_STRING = "tcp:127.0.0.1:5762"
    BAUD_RATE = None
else:
    CONNECTION_STRING = "/dev/ttyACM0"  # Update with your Pixhawk serial port
    BAUD_RATE = 115200

# =========================
# HELPER FUNCTIONS
# =========================
def connect_vehicle():
    print("=" * 60)
    print("CONNECTING TO VEHICLE...")
    print("=" * 60)
    if BAUD_RATE:
        vehicle = connect(CONNECTION_STRING, baud=BAUD_RATE, wait_ready=True, timeout=60)
    else:
        vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=60)

    print("Connected ✅")
    print(f"Firmware: {vehicle.version}")
    print(f"Mode    : {vehicle.mode.name}")
    return vehicle

def wait_until_armable(vehicle):
    while not vehicle.is_armable:
        print("Waiting until vehicle is armable...")
        time.sleep(1)

def wait_for_gps(vehicle):
    while getattr(vehicle, "gps_0", None) and vehicle.gps_0.fix_type < 3:
        print(f"Waiting for GPS fix... GPS fix={vehicle.gps_0.fix_type} sats={vehicle.gps_0.satellites_visible}")
        time.sleep(1)

def arm_and_takeoff(vehicle, target_altitude):
    print("Setting mode to GUIDED...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(0.2)

    wait_until_armable(vehicle)
    wait_for_gps(vehicle)

    print("Arming...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.2)
    time.sleep(3)
    print(f"Taking off to {target_altitude} meters...")
    vehicle.simple_takeoff(target_altitude)
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {altitude:.2f}m")
        if altitude >= 0.95 * target_altitude:
            break
        time.sleep(1)
    time.sleep(2)

def send_position_command(vehicle, lat, lon, alt):
    print(f"Moving to GPS location: ({lat}, {lon}, {alt})")
    target_location = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(target_location)

def descend_to_altitude(vehicle, target_altitude):
    print(f"Descending to {target_altitude} meters...")
    vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, target_altitude))
    while abs(vehicle.location.global_relative_frame.alt - target_altitude) > 1:
        print(f"Current altitude: {vehicle.location.global_relative_frame.alt}m")
        time.sleep(1)
    print(f"Reached target altitude of {target_altitude} meters.")

def land(vehicle):
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(f"Altitude: {vehicle.location.global_relative_frame.alt:.2f}m")
        time.sleep(1)
    print("Disarmed ✅")

def stop(vehicle):
    print("Stopping for safety.")
    vehicle.mode = VehicleMode("GUIDED")  # Keep it in GUIDED mode for further control
    vehicle.armed = False

# =========================
# MAIN SEQUENCE
# =========================
def main():
    vehicle = None
    try:
        vehicle = connect_vehicle()
        arm_and_takeoff(vehicle, TARGET_ALTITUDE)

        # Move to each target GPS location
        for target in GPS_TARGETS:
            lat, lon, alt = target
            send_position_command(vehicle, lat, lon, alt)
            
            # Wait until the drone reaches the target position (with tolerance)
            while abs(vehicle.location.global_relative_frame.lat - lat) > 0.0001 or \
                  abs(vehicle.location.global_relative_frame.lon - lon) > 0.0001 or \
                  abs(vehicle.location.global_relative_frame.alt - alt) > 5.0:
                print(f"Current position: {vehicle.location.global_relative_frame}")
                time.sleep(1)

        # Once mission is completed, descend to 5m before landing
        print("Mission complete, preparing to descend to 5 meters...")
        descend_to_altitude(vehicle, 5.0)  # Descend to 5 meters before landing

        print("All waypoints reached. Landing...")
        land(vehicle)

    except KeyboardInterrupt:
        print("\nEmergency: Land!")
        if vehicle:
            stop(vehicle)
    finally:
        if vehicle:
            vehicle.close()
            print("Closed the vehicle.")

if __name__ == "__main__":
    main()
