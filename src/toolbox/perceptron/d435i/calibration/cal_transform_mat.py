import socket
import numpy as np
import pyrealsense2 as rs
import cv2
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.toolbox.perceptron.d435i.realsense_depth import *

class CalibrationSystem:
    def __init__(self):
        # Socket setup for robot communication
        self.server_ip = '192.168.0.10'
        self.server_port = 10003
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))

        # Initialize depth camera
        self.dc = DepthCamera()
        time.sleep(3)  # Wait for camera to initialize

        # Storage for calibration data
        self.calibration_data = {
            'camera_points': [],
            'robot_points': [],
            'images': [],
            'depth_images': [],
            'intrinsics': []
        }

    def get_ee_pose(self):
        """Get end effector pose from robot"""
        message = "ReadActPos,0,;"
        self.client_socket.sendall(message.encode())
        response = self.client_socket.recv(1024).decode()
        response = response.split(',')
        return np.array([float(response[8]), float(response[9]), float(response[10]), 
                        float(response[11]), float(response[12]), float(response[13])])

    def get_camera_coordinate(self, depth_image, intrinsics_dict, x, y):
        """Convert pixel coordinates to 3D point in camera frame"""
        y_int = min(max(0, int(round(y))), depth_image.shape[0] - 1)
        x_int = min(max(0, int(round(x))), depth_image.shape[1] - 1)
        
        depth = depth_image[y_int, x_int].astype(float)
        if depth == 0:
            return None
        
        point = rs.rs2_deproject_pixel_to_point(intrinsics_dict, [x, y], depth)
        return point

    def get_ee_coordinate_in_cam(self, img, target_x_number, target_y_number):
        """Get end effector coordinates in camera frame"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (target_x_number, target_y_number), None)
        if not ret:
            return None, None
        
        corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:, i] = corners[i, 0, :]
        x = np.mean(corner_points[0])
        y = np.mean(corner_points[1])
        return x, y

    def estimate_transform(self):
        """Estimate transformation matrix between camera and robot frames"""
        camera_points = np.array(self.calibration_data['camera_points'])
        robot_points = np.array(self.calibration_data['robot_points'])
        robot_points = robot_points[:, :3]

        # Compute centroids
        centroid_camera = np.mean(camera_points, axis=0)
        centroid_robot = np.mean(robot_points, axis=0)

        # Center the points
        centered_camera = camera_points - centroid_camera
        centered_robot = robot_points - centroid_robot

        # Compute the covariance matrix
        H = np.dot(centered_camera.T, centered_robot)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R_estimated = np.dot(Vt.T, U.T)

        # Ensure proper rotation matrix
        if np.linalg.det(R_estimated) < 0:
            Vt[-1, :] *= -1
            R_estimated = np.dot(Vt.T, U.T)

        # Compute translation
        t_estimated = centroid_robot - np.dot(R_estimated, centroid_camera)

        # Create transformation matrix
        T_camera_to_base = np.eye(4)
        T_camera_to_base[:3, :3] = R_estimated
        T_camera_to_base[:3, 3] = t_estimated

        return T_camera_to_base

    def collect_sample(self):
        """Collect one sample of calibration data"""
        ret, depth_image, color_image, depth_frame, color_frame = self.dc.get_frame()
        if not ret:
            return False

        # Get robot pose
        robot_pose = self.get_ee_pose()
        
        # Get camera intrinsics
        depth_profile = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        # Store the data
        self.calibration_data['images'].append(color_image)
        self.calibration_data['depth_images'].append(depth_image)
        self.calibration_data['robot_points'].append(robot_pose)
        self.calibration_data['intrinsics'].append(depth_profile)
        
        return True
    
    def collect_and_process_sample(self, target_x_number, target_y_number):
        """Collect and process one sample of calibration data"""
        if not self.collect_sample():
            return False

        # Process the collected sample immediately
        latest_idx = len(self.calibration_data['images']) - 1
        x, y = self.get_ee_coordinate_in_cam(
            self.calibration_data['images'][latest_idx],
            target_x_number,
            target_y_number
        )
        
        if x is None:
            # If processing fails, remove the collected sample
            self.calibration_data['images'].pop()
            self.calibration_data['depth_images'].pop()
            self.calibration_data['robot_points'].pop()
            self.calibration_data['intrinsics'].pop()
            print("Failed to detect checkerboard pattern. Please try again.")
            return False
            
        camera_point = self.get_camera_coordinate(
            self.calibration_data['depth_images'][latest_idx],
            self.calibration_data['intrinsics'][latest_idx],
            x, y
        )
        if camera_point is None or np.allclose(camera_point, 0):
            # If point cloud processing fails, remove the collected sample
            self.calibration_data['images'].pop()
            self.calibration_data['depth_images'].pop()
            self.calibration_data['robot_points'].pop()
            self.calibration_data['intrinsics'].pop()
            print("Failed to get 3D coordinates. Please try again.")
            return False
        
        self.calibration_data['camera_points'].append(camera_point)
        return True
    
    def draw_checkerboard(self, image, corners, pattern_found, target_x_number, target_y_number):
        """Draw checkerboard corners and center point on the image"""
        display_image = image.copy()
        
        if pattern_found:
            # Draw the checkerboard corners
            cv2.drawChessboardCorners(display_image, (target_x_number, target_y_number), corners, pattern_found)
            
            # Calculate and draw the center point
            corner_points = corners.reshape(-1, 2)
            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))
            
            # Draw center point with a more visible design
            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)  # Red center
            cv2.circle(display_image, (center_x, center_y), 8, (255, 255, 255), 2)  # White outline
            
            # Add text to show pattern detection
            cv2.putText(display_image, "Pattern Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Add text when pattern is not detected
            cv2.putText(display_image, "No Pattern Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return display_image

    def calibrate(self, num_samples=10, target_x_number=9, target_y_number=6):
        """Run the complete calibration process"""
        print("Starting calibration. Press SPACE to collect samples, ESC to finish, d to delete the last sample, r to reset the calibration.")
        
        # Create and configure window
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', 640, 480)
        
        sample_count = 0
        while sample_count < num_samples:
            ret, depth_image, color_image, depth_frame, color_frame = self.dc.get_frame()
            if ret:
                # Detect checkerboard for visualization
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                pattern_found, corners = cv2.findChessboardCorners(gray, (target_x_number, target_y_number), None)
                
                if pattern_found:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw checkerboard visualization
                display_image = self.draw_checkerboard(color_image, corners, pattern_found, 
                                                     target_x_number, target_y_number)
                
                # Display sample counter
                cv2.putText(display_image, f"Samples: {sample_count}/{num_samples}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Camera Feed', display_image)
                key = cv2.waitKey(1)
                
                if key == 32:  # Space key
                    if self.collect_and_process_sample(target_x_number, target_y_number):
                        print(f"Collected and processed sample {sample_count + 1}/{num_samples}")
                        sample_count += 1
                    else:
                        print("Sample collection failed. Try again.")
                elif key == 27:  # ESC key
                    break
                elif key == 100:  # d key delete the last sample
                    if sample_count > 0:
                        self.calibration_data['images'].pop()
                        self.calibration_data['depth_images'].pop()
                        self.calibration_data['robot_points'].pop()
                        self.calibration_data['intrinsics'].pop()
                        self.calibration_data['camera_points'].pop()
                        print(f"Deleted sample {sample_count}/{num_samples}")
                        sample_count -= 1
                elif key == 114:  # r key reset the calibration
                    self.calibration_data['images'] = []
                    self.calibration_data['depth_images'] = []
                    self.calibration_data['robot_points'] = []
                    self.calibration_data['intrinsics'] = []
                    self.calibration_data['camera_points'] = []
                    print("Calibration data reset.")
                    sample_count = 0

        cv2.destroyAllWindows()

        # Calculate transformation matrix
        if len(self.calibration_data['camera_points']) > 0:
            transform_matrix = self.estimate_transform()
            print("Calibration completed. Transformation matrix:")
            print(',\n'.join(['[' + ', '.join([f'{x:10.6f}' for x in row]) + ']' for row in transform_matrix]))

            return transform_matrix
        else:
            print("No valid calibration points found!")
            return None

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.client_socket.close()
        self.dc.release()

if __name__ == "__main__":
    calibration_system = CalibrationSystem()
    print("Calibration started")
    num_samples = input("Input the number of samples: ")
    transform_matrix = calibration_system.calibrate(num_samples=int(num_samples)) 