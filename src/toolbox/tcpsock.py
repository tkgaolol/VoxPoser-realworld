import socket
import time

class TCP_SOCKET:
    """
    TCP socket for sending control message to the server
    """
    def __init__(self):
        """Initialize TCP socket with safety configurations"""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(2)
            # Enable TCP keepalive
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # Set TCP keepalive parameters (if supported by OS)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.tcp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.tcp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.tcp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        except socket.error as e:
            print(f'[ERROR]: Failed to initialize socket: {e}')
            raise

    def validate_connection_params(self, server_ip: str, server_port: int) -> bool:
        """Validate connection parameters"""
        try:
            # Validate IP address format
            socket.inet_aton(server_ip)
            # Validate port range
            if not (0 < server_port < 65536):
                raise ValueError(f"Invalid port number: {server_port}")
            return True
        except socket.error:
            print(f'[ERROR]: Invalid IP address: {server_ip}')
            return False
        except ValueError as e:
            print(f'[ERROR]: {e}')
            return False

    def connect(self, server_ip: str, server_port: int, max_retries: int = 5) -> bool:
        """Connect to server with retry mechanism and validation"""
        if not self.validate_connection_params(server_ip, server_port):
            return False

        server_address = (server_ip, server_port)
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.tcp_socket.connect(server_address)
                print(f'[INFO]: Successfully connected to {server_ip}:{server_port}')
                return True
            except socket.error as e:
                retry_count += 1
                print(f'[INFO]: Connection attempt {retry_count}/{max_retries} failed')
                print(f'[ERROR]: {e}')

                if retry_count < max_retries:
                    self.tcp_socket.close()
                    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.tcp_socket.settimeout(2)
                    time.sleep(5)
                else:
                    print('[ERROR]: Max connection retries reached')
                    return False
        return False

    def send_tcp_request(self, server_ip: str, server_port: int, message: str) -> tuple[bool, bytes]:
        """Send TCP request
        
        Returns:
            tuple: (success_state: bool, response: bytes)
        """
        # Validate input parameters
        if not isinstance(message, str) or not message:
            print("[ERROR]: Message must be a non-empty string")
            return False, b''
        
        server_port = int(server_port)

        try:
            # Check connection status
            try:
                current_peer = self.tcp_socket.getpeername()
                if current_peer != (server_ip, server_port):
                    print(f'[INFO]: Connection lost, reconnecting to {server_ip}:{server_port}')
                    if not self.connect(server_ip, server_port):
                        return False, b''
            except socket.error:
                print(f'[INFO]: Building connection to {server_ip}:{server_port}')
                if not self.connect(server_ip, server_port):
                    return False, b''

            print(f'[INFO]: Sending message: {message}')
            data = message.encode()
            self.tcp_socket.sendall(data)

            # Receive response with retry mechanism
            retry_count = 0
            max_retries = 10
            while retry_count <= max_retries:
                try:
                    response = self.tcp_socket.recv(1024)
                    print(f'[INFO]: Received from server: {response}')
                    return True, response
                except socket.timeout:
                    retry_count += 1
                    print(f'[WARNING]: Receive timeout (attempt {retry_count}/{max_retries})')
                    if retry_count <= max_retries:
                        self.tcp_socket.sendall(data)
                        time.sleep(1)
                        print('[INFO]: Resending message')
                    else:
                        print('[ERROR]: Max receive retries reached')
                        return False, b''
            return False, b''
        except Exception as e:
            print(f'[ERROR]: Failed to send TCP request: {e}')
            return False, b''

    # Close the connection
    def close_socket(self):
        """Safely close the socket connection"""
        try:
            if self.tcp_socket:
                self.tcp_socket.shutdown(socket.SHUT_RDWR)
                self.tcp_socket.close()
                print('[INFO]: Socket closed successfully')
        except socket.error as e:
            print(f'[ERROR]: Error closing socket: {e}')