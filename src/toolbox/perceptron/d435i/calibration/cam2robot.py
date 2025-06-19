import cv2
import pyrealsense2 as rs
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.toolbox.perceptron.d435i.realsense_depth import *

class CoordinateTransformer:
    def __init__(self, transform_matrix):
        """
        Initialize with the calibration transformation matrix
        Args:
            transform_matrix: 4x4 homogeneous transformation matrix from camera to robot base frame
        """
        self.T_camera_to_base = transform_matrix
        
    def camera_to_robot(self, point_camera):
        """
        Transform points from camera coordinates to robot base coordinates
        Args:
            point_camera: Array of shape (N, 3) containing N points in camera frame,
                         or single point of shape (3,)
        Returns:
            Array of shape (N, 3) containing points in robot base frame,
            or single point of shape (3,)
        """
        # Handle single point case
        single_point = len(point_camera.shape) == 1
        if single_point:
            point_camera = point_camera.reshape(1, 3)
        
        # Convert to homogeneous coordinates (N, 4)
        points_homog = np.hstack([point_camera, np.ones((point_camera.shape[0], 1))])
        
        # Apply transformation (N, 4)
        points_robot_homog = np.dot(points_homog, self.T_camera_to_base.T)
        
        # Convert back to 3D coordinates (N, 3)
        points_robot = points_robot_homog[:, :3] / points_robot_homog[:, 3:]
        
        return points_robot[0] if single_point else points_robot
    
        # # Convert to homogeneous coordinates
        # point_homog = np.append(point_camera, 1)
        
        # # Apply transformation
        # point_robot_homog = np.dot(self.T_camera_to_base, point_homog)
        
        # # Convert back to 3D coordinates
        # point_robot = point_robot_homog[:3] / point_robot_homog[3]
        
        # return point_robot

    def robot_to_camera(self, point_robot):
        """
        Transform points from robot base coordinates to camera coordinates
        Args:
            point_robot: Array of shape (N, 3) containing N points in robot base frame,
                        or single point of shape (3,)
        Returns:
            Array of shape (N, 3) containing points in camera frame,
            or single point of shape (3,)
        """
        # Handle single point case
        single_point = len(point_robot.shape) == 1
        if single_point:
            point_robot = point_robot.reshape(1, 3)
        
        # Convert to homogeneous coordinates (N, 4)
        points_homog = np.hstack([point_robot, np.ones((point_robot.shape[0], 1))])
        
        # Compute inverse transformation
        T_base_to_camera = np.linalg.inv(self.T_camera_to_base)
        
        # Apply transformation (N, 4)
        points_camera_homog = np.dot(points_homog, T_base_to_camera.T)
        
        # Convert back to 3D coordinates (N, 3)
        points_camera = points_camera_homog[:, :3] / points_camera_homog[:, 3:]
        
        return points_camera[0] if single_point else points_camera

        # # Convert to homogeneous coordinates
        # point_homog = np.append(point_robot, 1)
        
        # # Compute inverse transformation
        # T_base_to_camera = np.linalg.inv(self.T_camera_to_base)
        
        # # Apply transformation
        # point_camera_homog = np.dot(T_base_to_camera, point_homog)
        
        # # Convert back to 3D coordinates
        # point_camera = point_camera_homog[:3] / point_camera_homog[3]
        
        # return point_camera

point = (119, 265)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)
    print(x, y)

def get_pointcloud(depth_image, depth_intrinsics, x, y):
    depth = depth_image[y, x].astype(float)
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return point

if __name__ == "__main__":
    # Example transformation matrix (replace with your calibrated matrix)
    T_camera_to_base = np.array([
    [  0.771194,  -0.462891,   0.437026, -455.106040],
    [ -0.636553,  -0.552384,   0.538212, -947.105010],
    [ -0.007728,  -0.693256,  -0.720650, 578.839264],
    [  0.000000,   0.000000,   0.000000,   1.000000]
    ])
    
    # Create transformer instance
    transformer = CoordinateTransformer(T_camera_to_base)

    dc = DepthCamera()
    time.sleep(3)

    cv2.namedWindow("color image")
    cv2.setMouseCallback("color image", show_distance)

    while True:
        ret, depth_image, color_image, depth_frame, color_frame = dc.get_frame()
        
        cv2.circle(color_image, point, 4, (0, 0, 255))
        distance = depth_image[point[1], point[0]]

        cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        depth_profile = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        point_3d = get_pointcloud(depth_image, depth_profile, point[0], point[1])
        print(f"3D Point at ({point[0]}, {point[1]}): {point_3d}")

        point_in_camera = np.array(point_3d)  # Point in camera coordinates
        print("Point in camera frame:", point_in_camera)
        
        # Transform to robot base frame
        point_in_robot = transformer.camera_to_robot(point_in_camera)
        print("Point in robot base frame:", point_in_robot)
        
        # Transform back to camera frame to verify
        point_back_in_camera = transformer.robot_to_camera(point_in_robot)
        print("Point back in camera frame:", point_back_in_camera)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_RAINBOW)
        cv2.imshow("depth image", depth_colormap)
        cv2.imshow("color image", color_image)
        key = cv2.waitKey(1)
        if key == 27:
            break