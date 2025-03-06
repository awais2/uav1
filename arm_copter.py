#!/usr/bin/env python3

import time
from pymavlink import mavutil

def connect_mavlink(device='/dev/ttyUSB0', baudrate=57600):
    """
    Connect to the vehicle using MAVLink
    """
    print(f"Connecting to {device} at {baudrate} baud...")
    connection = mavutil.mavlink_connection(
        device,
        baud=baudrate,
    )
    
    # Wait for the first heartbeat to know the connection is established
    print("Waiting for heartbeat...")
    connection.wait_heartbeat()
    print(f"Heartbeat from system (system {connection.target_system}, component {connection.target_component})")
    
    return connection

def arm_vehicle(connection, force=False):
    """
    Arms the vehicle using MAVLink COMMAND_LONG message
    force=True will attempt to bypass arming checks
    """
    if force:
        print("Attempting to FORCE arm vehicle...")
        param2 = 21196  # Force arming parameter
    else:
        print("Attempting to arm vehicle normally...")
        param2 = 1  # Normal arming parameter
    
    # Format: target_system, target_component, command, confirmation, param1, param2, ...
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Command ID (400)
        0,  # Confirmation
        param2,  # param1: 1 to arm
        param2,  # param2: 21196 to force arm, otherwise 0
        0, 0, 0, 0, 0  # Remaining params
    )
    
    # Wait for command ACK
    ack = connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    if ack:
        if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("Vehicle armed successfully!")
            return True
        else:
            print(f"Arming failed with result: {ack.result}")
            return False
    else:
        print("No acknowledgment received from vehicle")
        return False

def disarm_vehicle(connection, force=False):
    """
    Disarms the vehicle using MAVLink COMMAND_LONG message
    force=True will attempt to force disarm even if flying
    """
    if force:
        print("Attempting to FORCE disarm vehicle...")
        param2 = 21196  # Force disarming parameter
    else:
        print("Attempting to disarm vehicle normally...")
        param2 = 0  # Normal disarming parameter
    
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Command ID (400)
        0,  # Confirmation
        0,  # param1: 0 to disarm
        param2,  # param2: 21196 to force disarm, otherwise 0
        0, 0, 0, 0, 0  # Remaining params
    )
    
    # Wait for command ACK
    ack = connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
    if ack:
        if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("Vehicle disarmed successfully!")
            return True
        else:
            print(f"Disarming failed with result: {ack.result}")
            return False
    else:
        print("No acknowledgment received from vehicle")
        return False

def main():
    # Connect to the vehicle
    connection = connect_mavlink()
    
    while True:
        print("\nUAV Control Menu:")
        print("1: Arm")
        print("2: Force Arm")
        print("3: Disarm")
        print("4: Force Disarm")
        print("5: Exit")
        
        action = input("Select action: ")
        
        if action == '1':
            arm_vehicle(connection, force=False)
        elif action == '2':
            arm_vehicle(connection, force=True)
        elif action == '3':
            disarm_vehicle(connection, force=False)
        elif action == '4':
            disarm_vehicle(connection, force=True)
        elif action == '5':
            break
        else:
            print("Invalid selection")
    
    # Close the connection
    print("Closing connection")
    connection.close()

if __name__ == "__main__":
    main()