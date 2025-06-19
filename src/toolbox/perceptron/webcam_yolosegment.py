from ultralytics import YOLO
import cv2

model = YOLO("src/model_weight/yolo11x-seg.pt")


# show the webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("image")

while True:
    ret, image = cap.read()

    if ret:
        results = model(image)
        print(results[0].boxes.xyxy)
        cv2.imshow("image", results[0].plot())
        key = cv2.waitKey(1)
        if key == 27:
            break

