from ultralytics import SAM
import cv2

# Load a model
model = SAM("src/model_weight/sam2.1_b.pt")
model.to("cuda")
# Display model information (optional)
model.info()

cap = cv2.VideoCapture(0)
cv2.namedWindow("image")

while True:
    ret, image = cap.read()
    if ret:
        # Run inference with bboxes prompt
        results = model(source=image, bboxes=[267.9727, 355.7712, 445.7959, 479.7076])
        if results:
            print(results[0].boxes.xyxy)
            cv2.imshow("image", results[0].plot())
            key = cv2.waitKey(1)
            if key == 27:
                break

# # Run inference with single point
# results = model(points=[900, 370], labels=[1])

# # Run inference with multiple points
# results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

# # Run inference with multiple points prompt per object
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# # Run inference with negative points prompt
# results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])