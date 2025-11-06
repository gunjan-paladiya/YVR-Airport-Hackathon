import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime, timedelta
import os
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Mapping from priority codes to colors
# PRIORITY_COLORS = {
#     "P1": (0, 0, 255),  # Red
#     "P2": (0, 255, 0),  # Green
#     "P3": (255, 255, 0)  # Yellow
# }


PRIORITY_COLORS ={
    "P1": (0, 0, 255),  # Red
    "P2": (255, 255, 0),  # Green
    "P3": (0, 255, 0)  # Yellow
}


def check_unattended_objects(unattended_objects, class_id, timestamp):
    current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    class_name = model.model.names[class_id]
    for obj in unattended_objects[class_name]:
        if (current_time - obj['last_seen']).seconds >= 10:
            return True
    return False

def get_priority(unattended_objects):
    if len(unattended_objects) == 0:
        return None
    last_seen_times = [obj['last_seen'] for objs in unattended_objects.values() for obj in objs]
    if not last_seen_times:
        return None  # No last_seen times available
    last_seen = max(last_seen_times)
    time_difference = (datetime.now() - last_seen).seconds
    if time_difference >= 30:
        return "P1"
    elif time_difference >= 20:
        return "P2"
    elif time_difference >= 10:
        return "P3"
    return None


def process_frame(frame, model, box_annotator, unattended_objects):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Generate timestamp here
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
    
    for i in range(len(detections)):
        class_id = detections.class_id[i]
        if not check_unattended_objects(unattended_objects, class_id, timestamp):  # Pass timestamp here
            # Check if there is at least one unattended object
            if any(unattended_objects.values()):
                # Change color based on priority
                priority = get_priority(unattended_objects)
                if priority in PRIORITY_COLORS:
                    color = PRIORITY_COLORS[priority]
                else:
                    color = (255, 255, 255)  # White for other cases
                bbox = detections.xyxy[i]  # Get bounding box coordinates directly
                annotated_frame = cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    return annotated_frame, detections, labels, model

# Rest of your code remains unchanged...
def default_serializer(obj):
    if isinstance(obj, (datetime, type(None))):
        return obj.strftime('%Y-%m-%d %H:%M:%S') if obj else None
    raise TypeError("Type not serializable")


def main(model):
    cap = None 
    try:
        args = parse_arguments()
        config = read_config(args.config)
        
        camera = config['cameras'][0]  # Assuming only one camera for simplicity
        rtsp_url = camera['rtsp_url']

        credentials = rtsp_url.split("://")[1].split("@")[0].split(":")
        username = credentials[0]
        password = credentials[1]

        print("Opening the camera...")
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Setting up annotator...")
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        output_data = []
        class_counts = defaultdict(dict)
        unattended_objects = defaultdict(list)

        terminal_output = []

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            print("Processing frame...")
            annotated_frame, detections, labels, model = process_frame(frame, model, box_annotator, unattended_objects)

            cv2.imshow("yolov9", annotated_frame)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "timestamp": timestamp,
                "detections": [{
                    "class_name": model.model.names[class_id],
                    "confidence": float(confidence)
                } for class_id, confidence in zip(detections.class_id, detections.confidence)]
            }

            output_data.append(data)

            unique_detections = set()
            for detection in data["detections"]:
                timestamp = data["timestamp"]
                class_id = None
                for key, value in model.model.names.items():
                    if value == detection["class_name"]:
                        class_id = key
                        break

                if class_id is not None:
                    unique_detections.add(class_id)
                    
                    if timestamp not in class_counts:
                        class_counts[timestamp] = {}
                    
                    class_counts[timestamp][detection["class_name"]] = class_counts[timestamp].get(detection["class_name"], 0) + 1

                    if 14 <= class_id <= 55 or 62 <= class_id <= 79:
                        unattended = check_unattended_objects(unattended_objects, class_id, timestamp)
                        if not unattended:
                            unattended_objects[model.model.names[class_id]].append({
                                'last_seen': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            })

            # Remove unattended objects if person is detected
            if 'person' in [d["class_name"] for d in data["detections"]]:
                for class_name in unattended_objects.keys():
                    unattended_objects[class_name] = [obj for obj in unattended_objects[class_name] if (datetime.now() - obj['last_seen']).seconds < 10]

            priority = get_priority(unattended_objects)
            if priority:
                print(f"Priority: {priority}")

            if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
                break

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

            

      


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv9 live")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    return args

def read_config(file_path: str):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config




if __name__ == "__main__":
    model = YOLO('yolov9e.pt')
    main(model)




