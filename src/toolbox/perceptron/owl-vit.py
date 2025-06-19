import requests
from PIL import Image
import torch
import numpy as np
import cv2

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import io
import matplotlib.pyplot as plt
import sys
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map=device)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", device_map=device)
# processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-finetuned")
# model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-finetuned", device_map=device)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
cap = cv2.VideoCapture(0)
time.sleep(3)
ret, image = cap.read()
# image = Image.open(requests.get(url, stream=True).raw)




# transform webcam to PIL Image and back to numpy array for drawing
image_pil = Image.fromarray(image)
buffer = io.BytesIO()
image_pil.save(buffer, format="JPEG")
buffer.seek(0)
image_pil = Image.open(buffer)
print(type(image_pil))


# texts = [["human","a photo of a tissue"]]
texts = [["red ball", "green ball"]]
inputs = processor(text=texts, images=image_pil, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image_pil.size[::-1]])
results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=texts)
result = results[0]
boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

# Convert PIL Image back to numpy array for drawing
image_draw = np.array(image_pil)

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {texts[0][label]} with confidence {round(score.item(), 3)} at location {box}")
    # Draw rectangle on numpy array
    cv2.rectangle(image_draw, 
                 (int(box[0]), int(box[1])), 
                 (int(box[2]), int(box[3])), 
                 (0, 0, 255), 
                 2)

fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((480, 640, 3)))
plt.title("Object Tracking")
ax.axis('off')  # Turn off axes
fig.tight_layout()  # Adjust layout to remove padding

img_display.set_array(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.draw()
plt.pause(10000000)
cap.release()
plt.close()
