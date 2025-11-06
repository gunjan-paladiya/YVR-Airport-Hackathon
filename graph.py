import cv2
import argparse
import json
from datetime import datetime, timedelta
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def process_frame(frame, model, box_annotator):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels)

    return annotated_frame, detections, labels, model


# Function to update the plot
def update_plot(frame):
    global output_data, class_counts_by_timestamp
    if frame < len(output_data):
        timestamp = datetime.strptime(output_data[frame]["timestamp"], "%Y-%m-%d %H:%M:%S")
        for detection in output_data[frame]["detections"]:
            class_name = detection["class_name"]
            class_counts_by_timestamp[class_name][timestamp] += 1

        ax.clear()
        for class_name, counts in class_counts_by_timestamp.items():
            timestamps = sorted(counts.keys())
            class_counts = [counts[ts] for ts in timestamps]
            ax.plot(timestamps, class_counts, label=class_name)

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Count")
        ax.set_title("Count of Each Class Over Time")
        ax.legend()
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    else:
        ani.event_source.stop()

# Load data from JSON file
with open("output.json", "r") as f:
    output_data = json.load(f)

# Process the data to get counts of each class at each timestamp
class_counts_by_timestamp = defaultdict(lambda: defaultdict(int))

# Create a figure and axis for the graph
fig, ax = plt.subplots(figsize=(12, 8))

# Create the animation for the graph
ani = FuncAnimation(fig, update_plot, frames=len(output_data), interval=10000)

# Initialize YOLO model
model = YOLO('yolov9e.pt')

# OpenCV window for video feed
cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Feed", 800, 600)

# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    annotated_frame, _, _, _ = process_frame(frame, model, box_annotator)

    # Display the frame
    cv2.imshow("Video Feed", annotated_frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt.tight_layout()
plt.show()

