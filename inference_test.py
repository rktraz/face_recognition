# uses webcam!!
# simply run this using "python3 inference_test.py"

import cv2
from ultralytics import YOLO


CONF = 0.5 # confidence threshold

mapping = {0.0: "Artem", 1.0: "Roman"}

# load my YOLO model
model = YOLO("models/medium_dataset_v6.pt")
model.conf = CONF

# using webcam
cap = cv2.VideoCapture(0)
detected_ids = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # object detection
    results = model(frame)

    # Check if detections were made
    if results:
        detected_classes = []

        # Iterate over each detection
        for box in results[0].boxes:
            # Retrieve class and confidence
            cls = box.cls  # class
            conf = box.conf  # confidence score

            # Append to list if confidence is higher than threshold
            if conf > CONF:  # Adjust the threshold if needed
                detected_classes.append(cls)

        # Print the detected class IDs for the current frame
        if detected_classes:
            detected_classes = [mapping[val.item()] for val in detected_classes]
            detected_ids.append(detected_classes)
        print("Detected IDs:", detected_classes)
    
    # Visualizing annotated_frame
    annotated_frame = frame


    cv2.imshow("YOLOv8 Inference", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("іди нахуй")
print(detected_ids)

cap.release()
cv2.destroyAllWindows()
