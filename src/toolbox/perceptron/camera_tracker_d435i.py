import time
import threading
import copy
import cv2
import torch
import numpy as np
# from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from ultralytics import SAM, YOLO
import sys
import os
import open3d as o3d
import pyrealsense2 as rs
import matplotlib.pyplot as plt

pwd = os.getcwd()
sys.path.append(os.path.join(pwd, 'src/toolbox/perceptron/XMem'))
from model.network import XMem
from inference.inference_core import InferenceCore

class CameraTracker:
    def __init__(self):
        """
        Initialize camera tracker with RealSense camera and object detection models.
        
        Args:
        """
        # Initialize device
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        except:
            self.device = torch.device("cpu")

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        # Configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Configure alignment
        self.align = rs.align(rs.stream.color)
        
        # Start streaming
        self.profile = self.pipeline.start(config)
        time.sleep(3)

        # Initialize OWL-ViT model
        # self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        # self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map=self.device)
        self.owlvit_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=self.device)

        # Initialize SAM model
        self.sam_model = SAM("src/model_weight/sam2.1_b.pt")
        self.sam_model.to(self.device)

        # Initialize XMem
        self.xmem_config = {
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'max_mid_term_frames': 10,
            'min_mid_term_frames': 5,
            'max_long_term_elements': 10000,
            'num_prototypes': 128,
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'key_dim': 64,
            'value_dim': 512,
            'hidden_dim': 64,
        }

        self.xmem_model = XMem(self.xmem_config)
        weights = torch.load('src/model_weight/XMem.pth', map_location=self.device)
        self.xmem_model.load_weights(weights)
        self.xmem_model.eval()
        self.xmem_model.to(self.device)

        self.yolo_model = YOLO('src/model_weight/yolo11x.pt')

        # Initialize variables
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_depth_image = None
        self.latest_color_image = None
        self.latest_objects = {}
        self.tracking_dict = {}
        self.object_names = []
        self.object_names_in_cam = []
        self.num_of_object = 0
        self.latest_pointcloud = None
        
        # Threading control
        self.first_frame = True
        self.is_running = False
        self.tracking_thread = None
        self.objects_lock = threading.Lock()
        self.first_frame_processed = None
        self.pcd = o3d.geometry.PointCloud()

    def process_frames(self):
        """Process latest frames and update object detections"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            self.aligned_frames = self.align.process(frames)
            # Get depth and color frames
            self.latest_depth_frame = self.aligned_frames.get_depth_frame()
            self.latest_color_frame = self.aligned_frames.get_color_frame()
            
            if not self.latest_depth_frame or not self.latest_color_frame:
                return
            
            # Convert frames to numpy arrays
            self.latest_depth_image = np.asanyarray(self.latest_depth_frame.get_data())
            self.latest_color_image = np.asanyarray(self.latest_color_frame.get_data())

            with self.objects_lock:
                input_image = cv2.cvtColor(self.latest_color_image, cv2.COLOR_BGR2RGB)
                if self.first_frame:
                    # Initialize tracking for all objects
                    object_names = [list(self.object_names)]
                    
                    try:
                        # Initialize detection
                        self.object_names_in_cam = []
                        filtered_boxes = []

                        for object_name in object_names[0]:  # Iterate over each object name
                            # Prepare inputs for the current object
                            inputs = self.owlvit_processor(text=[object_name], images=input_image, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                outputs = self.owlvit_model(**inputs)

                            target_sizes = torch.Tensor([input_image.shape[:2][::-1]]).to(self.device)
                            results = self.owlvit_processor.post_process_grounded_object_detection(
                                outputs=outputs, 
                                target_sizes=target_sizes, 
                                threshold=0.1, 
                                text_labels=[object_name]  # Use the current object name
                            )

                            boxes = results[0]["boxes"]
                            scores = results[0]["scores"]
                            labels = results[0]["labels"]

                            label_mask = labels == 0  # Assuming we are processing one object at a time
                            
                            if torch.any(label_mask):
                                mask_indices = torch.where(label_mask)[0]
                                label_scores = scores[label_mask]
                                best_filtered_idx = label_scores.argmax()
                                best_idx = mask_indices[best_filtered_idx]
                                best_score = label_scores[best_filtered_idx]
                                
                                if best_score > 0.1:
                                    filtered_boxes.append(boxes[best_idx].cpu().numpy())
                                    self.object_names_in_cam.append(object_name)  # Append the current object name

                        boxes = np.array(filtered_boxes)

                        if len(boxes) != 0:
                            sam_results = self.sam_model(source=input_image, bboxes=boxes)
                        else:
                            sam_results = None

                        if not sam_results or len(sam_results) == 0:
                            return

                        combined_mask = None
                        if sam_results is not None:
                            combined_mask = np.zeros((sam_results[0].masks.shape), dtype=np.float32)
                            for i in range(sam_results[0].masks.shape[0]):
                                mask = sam_results[0].masks.data[i].cpu().numpy()
                                combined_mask[i] = mask

                        frame_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).to(self.device) / 255
                        mask_tensor = torch.from_numpy(combined_mask).float().to(self.device)

                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                        processor = InferenceCore(self.xmem_model, self.xmem_config)
                        processor.all_labels = list(range(1, mask_tensor.shape[0] + 1))

                        with torch.no_grad():
                            processor.step(frame_tensor, mask_tensor)

                        self.tracking_dict = {
                            'processor': processor,
                        }

                        for object_name in self.object_names:
                            self.latest_objects[object_name] = {
                                'mask2d': None,
                                'mask3d': None,
                                'center3d': None,
                                'normal': None
                            }
                        
                        self.first_frame = False
                        
                        if hasattr(self, 'first_frame_processed'):
                            self.first_frame_processed.set()
                            
                    except Exception as e:
                        print(f"Error in first frame processing: {str(e)}")
                        self.first_frame = True
                        if hasattr(self, 'first_frame_processed'):
                            self.first_frame_processed.set()
                        return

                else:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    frame_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1).to(self.device) / 255

                    with torch.no_grad():
                        prob = self.tracking_dict['processor'].step(frame_tensor)

                    for i in range(1, len(self.object_names_in_cam) + 1):
                        mask = (prob[i] > 0.5).cpu().numpy()
                        self.latest_objects[self.object_names_in_cam[i-1]]['mask2d'] = mask

        except Exception as e:
            self.first_frame = True
            print(f"Error processing frames: {str(e)}")
    
    def process_3d(self):
        # Get 3D information using point cloud
        with self.objects_lock:
            depth_profile = self.latest_depth_frame.profile.as_video_stream_profile().get_intrinsics()
            pc = np.zeros((self.latest_depth_image.shape[0], self.latest_depth_image.shape[1], 3))  # Initialize with same size as Femtomega
            
            # Convert depth image to point cloud
            for x in range(self.latest_depth_image.shape[1]):
                for y in range(self.latest_depth_image.shape[0]):
                    depth = self.latest_depth_image[y, x].astype(float)
                    if depth > 0:
                        point = rs.rs2_deproject_pixel_to_point(depth_profile, [x, y], depth)
                        pc[y, x] = point

            self.latest_pointcloud = pc
            
            if pc is not None and bool(self.latest_objects):
                for object_name in self.object_names_in_cam:
                    mask2d = self.latest_objects[object_name]['mask2d']
                    if mask2d is None:
                        continue

                    # Calculate 2D center
                    y_indices, x_indices = np.where(mask2d)
                    if len(y_indices) > 0:
                        # centerx = int(np.mean(x_indices))
                        # centery = int(np.mean(y_indices))
                        
                        # # Get 3D center from point cloud
                        # center3d = pc[centery, centerx]


                        # Get masked point cloud
                        pc_masked = pc[mask2d]
                        # Remove zero points and outliers
                        pc_masked = pc_masked[~np.all(pc_masked == 0, axis=1)]  # Remove zero points
                        if len(pc_masked) > 0:
                            # Convert to open3d point cloud for outlier removal
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc_masked)
                            # Remove statistical outliers
                            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=250, std_ratio=2.0)
                            pc_masked = np.asarray(pcd.points)
                        center3d = np.mean(pc_masked, axis=0)

                        self.latest_objects[object_name]['mask3d'] = pc_masked
                        if not np.allclose(center3d, 0):
                            self.latest_objects[object_name]['center3d'] = center3d
                        else:
                            self.latest_objects[object_name]['center3d'] = np.zeros(3)

                        # Compute surface normal if enough points available
                        if len(pc_masked) >= 3:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc_masked)
                            pcd.estimate_normals()
                            normals = np.asarray(pcd.normals)
                            normal = np.mean(normals, axis=0)
                            normal = normal / np.linalg.norm(normal)
                            self.latest_objects[object_name]['normal'] = normal
                        else:
                            self.latest_objects[object_name]['normal'] = np.array([0, 0, 1])
                    else:
                        self.latest_objects[object_name]['center3d'] = np.zeros(3)
                        self.latest_objects[object_name]['normal'] = np.array([0, 0, 1])

    def visualize_pointcloud(self):
        """Visualize the latest point cloud with normals using Open3D"""
        for object_name in self.object_names:
            # Create point clouds for both cameras
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.latest_objects[object_name]['mask3d'])

            # Create a sphere at center3d position
            center = self.latest_objects[object_name]['center3d']
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.translate(center)
            sphere.paint_uniform_color([0, 0, 0])  # Black color

            # Create a line segment to represent the normal vector
            normal = self.latest_objects[object_name]['normal']
            normal_length = 10  # Length of the normal vector visualization
            line_points = np.array([center, center + normal * normal_length])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])  # Connect points 0 and 1
            line_set.paint_uniform_color([1, 0, 0])  # Red color for normal vector

            # Visualize point clouds, center sphere, and normal vector
            o3d.visualization.draw_geometries([pcd, sphere, line_set])

    def get_current_objects(self):
        """Use YOLO to get current objects"""
        frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(frames)
        self.latest_depth_frame = self.aligned_frames.get_depth_frame()
        self.latest_color_frame = self.aligned_frames.get_color_frame()
            
        if not self.latest_depth_frame or not self.latest_color_frame:
            return
            
        self.latest_depth_image = np.asanyarray(self.latest_depth_frame.get_data())
        self.latest_color_image = np.asanyarray(self.latest_color_frame.get_data())

        results = self.yolo_model(self.latest_color_image, verbose=False)
        
        current_objects = set()
        if results is not None:
            for result in results:
                if result.boxes is not None:
                    for i in result.boxes.cls:
                        current_objects.add(result.names[int(i)])
        return list(current_objects)

    def get_latest_frames(self):
        """Get the latest color and depth images"""
        return self.latest_color_image, self.latest_depth_image
        
    def get_latest_objects(self, object_names: list[str]=[], restart=False):
        """
        Get the latest detected objects. If new objects are provided, they will be added to tracking.
        
        Args:
            object_names: List of object names to track
        
        Returns:
            Dictionary containing latest object detections
        """
        with self.objects_lock:
            new_objects = False
            for name in object_names:
                if name not in self.object_names:
                    new_objects = True
                    break
                
            if new_objects or restart:
                # self.object_names = list(set(self.object_names + object_names))
                self.object_names = list(set(object_names))

                self.num_of_object = len(self.object_names)
                self.tracking_dict = {}
                self.latest_objects = {}
                self.first_frame = True
                self.first_frame_processed = threading.Event()
                self.start_tracking()

        if new_objects:
            if not self.first_frame_processed.wait(timeout=5.0):
                print("Warning: Timeout waiting for first frame processing")

        with self.objects_lock:
            return copy.deepcopy(self.latest_objects)

    def start_tracking(self):
        """Start the tracking thread"""
        if not self.is_running:
            self.is_running = True
            self.tracking_thread = threading.Thread(target=self.tracking_loop)
            self.tracking_thread.start()

    def tracking_loop(self):
        """Main tracking loop"""
        while self.is_running:
            self.process_frames()
            time.sleep(0.03)  # ~30 FPS

    def stop_tracking(self):
        """Stop the tracking thread and clean up"""
        self.is_running = False
        if self.tracking_thread:
            self.tracking_thread.join()
        self.pipeline.stop()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def reset(self):
        """Reset the camera tracker"""
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_depth_image = None
        self.latest_color_image = None
        self.latest_objects = {}
        self.tracking_dict = {}
        self.object_names = []
        self.object_names_in_cam = []
        self.num_of_object = 0
        self.latest_pointcloud = None
        
        self.first_frame = True
        self.is_running = False
        self.first_frame_processed = None

def on_key_press(tracker, event):
    """Handle keyboard events"""
    if event.key == 'q':
        plt.close('all')
        tracker.stop_tracking()
        sys.exit(0)
    elif event.key == 'r':
        tracker.reset()
        if tracker.device.type == 'cuda':
            torch.cuda.empty_cache()

    elif event.key == 'v':
        tracker.visualize_pointcloud()
        time.sleep(1000)
        sys.exit(0)

if __name__ == "__main__":
    # Initialize camera tracker
    print("Initializing Camera Tracker...")
    tracker = CameraTracker()

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    img_display = ax.imshow(np.zeros((480, 640, 3)))
    ax.set_title("Camera 1")
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(tracker, event))

    # Get object names from user
    num_objects = int(input("Enter the number of objects to track: "))
    object_names = []
    for i in range(num_objects):
        object_names.append(input(f"Enter the object name to track {i+1}: "))

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    count = 0

    try:
        while True:
            # Start tracking in background thread
            # count += 1
            # if count == 50:
            #     print("Switching to tracking 'ears'...")
            #     object_names = ['ears']

            dict_objects = tracker.get_latest_objects(object_names)
            
            # Skip visualization if no valid frames yet
            if tracker.latest_color_image is None:
                plt.pause(0.001)
                continue
                
            vis_frame = tracker.latest_color_image.copy()
            
            if dict_objects:
                tracker.process_3d()
                for i, object_name in enumerate(dict_objects):
                    # print(dict_objects[object_name])
                    color = colors[i % len(colors)]
                    print(f"{object_name} is {color}")
                    mask = dict_objects[object_name]['mask2d']
                    if mask is None:
                        continue
                    if mask is not None:
                        vis_frame[mask > 0] = vis_frame[mask > 0] * 0.7 + np.array(color, dtype=np.uint8) * 0.3
            
            img_display.set_array(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopping tracking...")
        tracker.stop_tracking()
        plt.close('all')