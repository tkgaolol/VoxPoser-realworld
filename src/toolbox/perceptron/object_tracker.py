import cv2
import torch
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from ultralytics import SAM
import sys
import os
import matplotlib.pyplot as plt

# Add the XMem folder to Python path
pwd = os.getcwd()
sys.path.append(os.path.join(pwd, 'src/toolbox/perceptron/XMem'))
from model.network import XMem
from inference.inference_core import InferenceCore

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
# OWL-ViT
owlvit_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=device)

# SAM
sam_model = SAM("src/model_weight/sam2.1_b.pt")
sam_model.to(device)

# XMem
xmem_config = {
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

xmem_model = XMem(xmem_config)
weights = torch.load('src/model_weight/XMem.pth', map_location=device)
xmem_model.load_weights(weights)
xmem_model.eval()
xmem_model.to(device)

def resize_frame(frame, target_size=480):
    h, w = frame.shape[:2]
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    return cv2.resize(frame, (new_w, new_h))

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Replace cv2.namedWindow with matplotlib setup
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((480, 640, 3)))
    plt.title("Object Tracking")
    ax.axis('off')  # Turn off axes
    fig.tight_layout()  # Adjust layout to remove padding

    # Get object name from user
    num_of_object = int(input("Enter the number of objects to track: "))
    object_name = [[]]
    for i in range(num_of_object):
        object_name[0].append(input(f"Enter the object name to track {i+1}: "))

    # Initialize variables
    tracking = False
    processor = None
    first_frame = None

    # Add keyboard event handler
    def on_key(event):
        if event.key == 'q':
            plt.close('all')
            cap.release()
            sys.exit(0)
        elif event.key == 'r':
            nonlocal tracking, processor, first_frame
            tracking = False
            processor = None
            first_frame = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for processing
            frame = resize_frame(frame)

            if not tracking:
                # Convert frame for OWL-ViT
                inputs = owlvit_processor(text=object_name, images=frame, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = owlvit_model(**inputs)

                # Process OWL-ViT results
                target_sizes = torch.Tensor([frame.shape[:2]]).to(device)
                results = owlvit_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=object_name)
                
                boxes = results[0]["boxes"]
                scores = results[0]["scores"]
                labels = results[0]["labels"]

                # Filter to keep only highest confidence detection per label
                filtered_boxes = []
                filtered_scores = []
                filtered_labels = []
                
                for i in range(num_of_object):
                    # Get indices where label matches current object
                    label_mask = labels == i
                    if torch.any(label_mask):
                        # Find index of highest scoring detection for this label
                        label_scores = scores[label_mask]
                        best_idx = label_scores.argmax()
                        
                        # Keep the best detection for this label
                        filtered_boxes.append(boxes[label_mask][best_idx].cpu().numpy())
                        filtered_scores.append(scores[label_mask][best_idx])
                        filtered_labels.append(i)
                
                # Update the original arrays with filtered results
                boxes = torch.stack([torch.from_numpy(box).to(device) for box in filtered_boxes])
                scores = torch.stack(filtered_scores)
                labels = torch.tensor(filtered_labels, device=device)

                boxes = boxes.cpu().numpy()
                # Get SAM mask using the box
                sam_results = sam_model(source=frame, bboxes=boxes)
                if sam_results and len(sam_results) > 0:
                    combined_mask = np.zeros((sam_results[0].masks.shape), dtype=np.float32)

                    for i in range(sam_results[0].masks.shape[0]):
                        mask = sam_results[0].masks.data[i].cpu().numpy()
                        combined_mask[i] = mask
                    
                    
                    # Initialize XMem tracking
                    first_frame = frame
                    frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).to(device) / 255
                    mask_tensor = torch.from_numpy(combined_mask).float().to(device)

                    # Clear GPU cache before starting new track
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Initialize the processor
                    processor = InferenceCore(xmem_model, xmem_config)
                    processor.all_labels = list(range(1, num_of_object + 1))  # Multi object tracking
                    
                    # Initialize tracking with all masks
                    with torch.no_grad():
                        processor.step(frame_tensor, mask_tensor)
                    tracking = True

            else:
                # Convert frame to tensor for XMem
                frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).to(device) / 255
                
                # Get the next mask
                with torch.no_grad():
                    prob = processor.step(frame_tensor)
                
                # Convert back to numpy for visualization
                masks = []
                for i in range(1, num_of_object + 1):
                    mask = (prob[i] > 0.5).cpu().numpy()
                    masks.append(mask)
                
                # Create visualization with different colors for each object
                vis_frame = frame.copy()
                colors = [
                    (255, 0, 0),  # Red
                    (0, 255, 0),  # Green
                    (0, 0, 255),  # Blue
                    (255, 255, 0),  # Yellow
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Cyan
                ]
                
                for i, mask in enumerate(masks):
                    color = colors[i % len(colors)]
                    vis_frame[mask > 0] = vis_frame[mask > 0] * 0.7 + np.array(color, dtype=np.uint8) * 0.3
                
                # Update matplotlib display
                img_display.set_array(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
                plt.draw()
                plt.pause(0.001)  # Add small pause to allow plot to update
            
            if not tracking:
                # Update matplotlib display for non-tracking state
                img_display.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.draw()
                plt.pause(0.001)

    finally:
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        cap.release()
        plt.ioff()  # Disable interactive mode
        plt.close()

if __name__ == "__main__":
    main() 