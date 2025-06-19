import serial
import time
import os

class GripperController:
    def __init__(self, port='COM3', baudrate=115200):
        """Initialize the gripper controller"""
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )

        # Set terminator to CR (Carriage Return)
        self.serial.write_terminator = '\r'
        self.serial.read_terminator = '\r'

        # For continuous reading mode, we can use non-blocking reads
        # self.serial.timeout = 0  # Non-blocking read
        
        time.sleep(3)  # Wait for serial connection to establish
        
        # Send initialization command
        init_cmd = [0x01, 0x06, 0x01, 0x00, 0x00, 0xA5]
        self._send_command(init_cmd)
        time.sleep(5)  # Wait for initialization to complete

        self.set_velocity(20)  # Set velocity to 20%
        time.sleep(0.5)
        self.set_force(20)     # Set force to 20%
        time.sleep(0.5)
        
    def _calculate_crc(self, data):
        """Calculate Modbus CRC16"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1

        crc_lo = crc & 0xFF
        crc_hi = (crc >> 8) & 0xFF
        return [crc_lo, crc_hi]

    def _send_command(self, command):
        """Send command with CRC"""
        cmd_bytes = bytes(command)
        crc = self._calculate_crc(cmd_bytes)
        command_with_crc = command + crc
        print(f"[INFO]: Sending GRIPPER command: {' '.join([f'{b:02X}' for b in command_with_crc])}")
        self.serial.write(bytes(command_with_crc))
        time.sleep(1)
        response = self.serial.read(len(command_with_crc))
        if response == bytes(command_with_crc):
            print("[INFO]: Command sent successfully")
            return None
        else:
            return response
        
    def set_position(self, position):
        """Set gripper position (0-1000) 00 00 – 03 E8"""
        if not 0 <= position <= 1000:
            raise ValueError("Position must be between 0 and 1000")
            
        print(f"[INFO]: Setting GRIPPER position to {position}")
        position_hex = f"{position:04X}"  # Convert to 4-digit hex
        cmd = [
            0x01,
            0x06,
            0x01,
            0x03,
            int(position_hex[:2], 16),
            int(position_hex[2:], 16)
        ]
        self._send_command(cmd)
        time.sleep(3)

    def set_force(self, force):
        """Set gripper force (20-100) 00 14 – 00 64"""
        if not 20 <= force <= 100:
            raise ValueError("Force must be between 20 and 100")
        
        print(f"[INFO]: Setting GRIPPER force to {force}")
        # convert force to four digit hex with int format
        force_hex = f"{force:04X}"
        cmd = [
            0x01,
            0x06,
            0x01,
            0x01,
            int(force_hex[:2], 16),
            int(force_hex[2:], 16)
        ]
        self._send_command(cmd)
        time.sleep(0.1)

    def set_velocity(self, velocity):
        """Set gripper velocity (1-100) 00 01 – 00 64"""
        if not 1 <= velocity <= 100:
            raise ValueError("Velocity must be between 1 and 100")
        
        print(f"[INFO]: Setting GRIPPER velocity to {velocity}")
        velocity_hex = f"{velocity:04X}"
        cmd = [
            0x01,
            0x06,
            0x01,
            0x04,
            int(velocity_hex[:2], 16),
            int(velocity_hex[2:], 16)
        ]
        self._send_command(cmd)
        time.sleep(0.1)

    def get_position(self):
        """Read current position"""
        print("[INFO]: Reading GRIPPER position")
        cmd = [0x01, 0x03, 0x02, 0x02, 0x00, 0x01]
        response = self._send_command(cmd)
        time.sleep(0.5)

        print(f"[INFO]: Received GRIPPER response: {' '.join([f'{b:02X}' for b in response])}")
        position = (response[3] << 8) + response[4]
        print(f"[INFO]: GRIPPER position: {position}")
        return position

    def close(self):
        """Close the serial connection"""
        self.serial.close()

# Example usage
def main():
    try:
        # check if os is linux or windows
        if os.name != 'nt':
            # sudo chmod 666 /dev/ttyUSB0
            port = '/dev/ttyUSB0'
        else:
            port = 'COM3'

        # Initialize gripper
        gripper = GripperController(port=port)

        gripper.set_position(970)
        time.sleep(1)
        
        position = gripper.get_position()
        print(f"Current gripper position: {position}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        gripper.close()

if __name__ == "__main__":
    main()