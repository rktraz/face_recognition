
import cv2
import time
from collections import Counter
from ultralytics import YOLO

CONF = 0.5  # Confidence threshold
DETECTION_INTERVAL = 3  # Detection interval in seconds
MIN_DETECTIONS = 5  # Minimum number of detections for a person
MIN_CONFIDENCE = 0.8  # Minimum confidence for a person

# Load your YOLO model
model = YOLO("models/medium_dataset_v6.pt")
model.conf = CONF

# Using webcam
cap = cv2.VideoCapture(0)
detected_ids = []
start_time = time.time()
person_counts = Counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
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
            if conf > CONF:
                detected_classes.append(cls.item())

        # Print the detected class IDs for the current frame
        if detected_classes:
            detected_ids.append(detected_classes)

            elapsed_time = time.time() - start_time

            # Check if the detection interval has passed
            if elapsed_time >= DETECTION_INTERVAL:
                # Flatten the list of detected classes
                flattened_detected_classes = [item for sublist in detected_ids for item in sublist]

                # Count the frequency of each detected class
                counts = Counter(flattened_detected_classes)

                # Reset counts and start time
                detected_ids = []
                start_time = time.time()

                # Check if conditions are met for any detected person
                for person_id, count in counts.items():
                    confidence = count / len(flattened_detected_classes)
                    if count >= MIN_DETECTIONS and confidence >= MIN_CONFIDENCE:
                        print(f"I see person {person_id}!")

    # Visualizing annotated_frame
    annotated_frame = frame

    cv2.imshow("YOLOv8 Inference", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()