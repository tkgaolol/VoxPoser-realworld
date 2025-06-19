import cv2
from src.toolbox.perceptron.d435i.realsense_depth import *
import numpy as np

point = (119, 265)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)
    print(x, y)

def get_pointcloud(depth_image, depth_intrinsics, x, y):
    depth = depth_image[y, x].astype(float)
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return point

# Initialize the camera
dc = DepthCamera()

# Create a window and attach a mouse callback
cv2.namedWindow("color image")
cv2.setMouseCallback("color image", show_distance)

while True:
    ret, depth_image, color_image, depth_frame, color_frame  = dc.get_frame()

 
    # Show distance for a specific point
    cv2.circle(color_image, point, 4, (0, 0, 255))
    distance = depth_image[point[1], point[0]]

    cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    
    # Get depth data array
    depth_profile = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    point_3d = get_pointcloud(depth_image, depth_profile, point[0], point[1])
    print(f"3D Point at ({point[0]}, {point[1]}): {point_3d}")
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_RAINBOW)
    # cv2.imshow("depth image", depth_image)
    cv2.imshow("depth image", depth_colormap)
    cv2.imshow("color image", color_image)
    key = cv2.waitKey(1)
    if key == 27:
        break
