import serial
import serial.tools.list_ports  # Add this import at the top

@staticmethod
def list_ports():
    """List all available COM ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found")
        return []
    
    available_ports = []
    for port in ports:
        print(f"Found {port.device}: {port.description}")
        available_ports.append(port.device)
    return available_ports


def main():
    list_ports()

if __name__ == "__main__":
    main()
