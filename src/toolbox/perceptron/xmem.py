import os
import cv2
import torch
import numpy as np
import os
pwd = os.getcwd()
print(pwd)

# Add the XMem folder to Python path
import sys
sys.path.append(os.path.join(pwd, 'src/toolbox/perceptron/XMem'))

# Now import with the correct path
from model.network import XMem
from inference.inference_core import InferenceCore

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set target size for processing
TARGET_SIZE = 480  # Resize shorter side to this size

# XMem configuration
config = {
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

# Load XMem model
model = XMem(config)
weights = torch.load('src/model_weight/XMem.pth', map_location=device)
model.load_weights(weights)
model.eval()
model.to(device)

# Enable mixed precision inference if CUDA is available
if device.type == 'cuda':
    amp_context = torch.cuda.amp.autocast()
    amp_context.__enter__()

# Initialize video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('XMem Tracking Demo')

# Initialize variables
roi_selected = False
tracking = False
first_frame = None
mask = None
processor = None

def resize_frame(frame):
    h, w = frame.shape[:2]
    if h < w:
        new_h = TARGET_SIZE
        new_w = int(w * (TARGET_SIZE / h))
    else:
        new_w = TARGET_SIZE
        new_h = int(h * (TARGET_SIZE / w))
    return cv2.resize(frame, (new_w, new_h))

def mouse_callback(event, x, y, flags, param):
    global roi_selected, tracking, first_frame, mask, processor
    if event == cv2.EVENT_LBUTTONDOWN and not roi_selected:
        # Start ROI selection
        param['start_pos'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and not roi_selected:
        # Complete ROI selection
        end_pos = (x, y)
        start_pos = param['start_pos']
        
        # Create initial mask
        mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
        x1, y1 = min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1])
        x2, y2 = max(start_pos[0], end_pos[0]), max(start_pos[1], end_pos[1])
        mask[y1:y2, x1:x2] = 1
        
        roi_selected = True
        tracking = True
        
        # Convert first frame to tensor
        first_frame_tensor = torch.from_numpy(first_frame).float().permute(2, 0, 1).to(device) / 255
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)
        
        # Clear GPU cache before starting new track
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Initialize the processor with the same config
        processor = InferenceCore(model, config)
        # Set labels for single object tracking
        processor.all_labels = [1]  # Single object with label 1
        # Initialize with first frame and mask
        with torch.no_grad():
            processor.step(first_frame_tensor, mask_tensor)

cv2.setMouseCallback('XMem Tracking Demo', mouse_callback, 
                    {'start_pos': None})

print("Click and drag to select an object to track. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for processing
        frame = resize_frame(frame)
            
        if first_frame is None:
            first_frame = frame
            
        if tracking and processor is not None:
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).to(device) / 255
            
            # Get the next mask
            with torch.no_grad():
                prob = processor.step(frame_tensor)
            
            # Convert back to numpy for visualization
            mask = (prob[1] > 0.5).cpu().numpy()  # Use index 1 since 0 is background
            
            # Visualize the mask
            vis_frame = frame.copy()
            vis_frame[mask > 0] = vis_frame[mask > 0] * 0.7 + np.array([0, 0, 255], dtype=np.uint8) * 0.3
            
            cv2.imshow('XMem Tracking Demo', vis_frame)
        else:
            cv2.imshow('XMem Tracking Demo', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset tracking
            roi_selected = False
            tracking = False
            first_frame = frame
            mask = None
            processor = None
            # Clear GPU cache when resetting
            if device.type == 'cuda':
                torch.cuda.empty_cache()

finally:
    # Clean up
    if device.type == 'cuda':
        amp_context.__exit__(None, None, None)
        torch.cuda.empty_cache()
    cap.release()
    cv2.destroyAllWindows() 