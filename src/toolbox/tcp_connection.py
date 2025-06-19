import socket

# Server details
server_ip = '192.168.0.10'  # Replace with the IP address of the server
server_port = 10003  # Replace with the port number the server is listening on

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
server_address = (server_ip, server_port)
client_socket.connect(server_address)

# Send data to the server
# GrpEnable
command = 'GrpEnable'
nRbtID = '0'
message = command+','+nRbtID+',;'

print(message)
client_socket.sendall(message.encode())

# Receive data from the server
data = client_socket.recv(1024)
print('Received from server:', data.decode())

# Close the connection
client_socket.close()