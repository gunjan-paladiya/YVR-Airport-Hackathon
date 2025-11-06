import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime, timedelta
import os
from ultralytics import YOLO
from collections import defaultdict

# Mapping from priority codes to colors
PRIORITY_COLORS ={
    "P1": (0, 0, 255),  # Red
    "P2": (255, 255, 0),  # Yellow
    "P3": (0, 255, 0)  # Green
}

# List of classes to detect
CLASSES_TO_DETECT = ["suitcase", "backpack"]

def check_unattended_objects(unattended_objects, class_id, timestamp):
    current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    for obj in unattended_objects[class_id]:
        if (current_time - obj['last_seen']) >= timedelta(seconds=10):
            return True
    return False

def get_priority(unattended_objects):
    if not unattended_objects:
        return None  # Return None if unattended_objects is empty
    last_seen_times = [obj['last_seen'] for obj in unattended_objects]
    last_seen = max(last_seen_times)
    time_difference = (datetime.now() - last_seen).seconds
    if time_difference >= 30:
        return "P1"
    elif time_difference >= 15:
        return "P2"
    elif time_difference >= 5:
        return "P3"
    return None

def process_frame(frame, model, box_annotator, unattended_objects, output_folder):
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
        labels=labels
    )

    for i in range(len(detections)):
        class_id = detections.class_id[i]
        class_name = model.model.names[class_id]
        if class_name in CLASSES_TO_DETECT:
            if check_unattended_objects(unattended_objects[class_name], class_name, timestamp):
                priority = get_priority(unattended_objects[class_name])
                if priority in PRIORITY_COLORS:
                    color = PRIORITY_COLORS[priority]
                else:
                    color = (0, 0, 255)  # Blue for other cases
                # Add transparency to the color
                color_with_alpha = color + (64,)  # Reduced alpha value for more transparency
                bbox = detections.xyxy[i]  # Get bounding box coordinates directly
                # Fill the rectangular area with the specified color and opacity
                annotated_frame = cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_with_alpha, -1)
                # Add text overlay for unattended object
                cv2.putText(annotated_frame, "Unattended", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Save the frame when an unattended object is detected
                cv2.imwrite(os.path.join(output_folder, f"{timestamp.replace(':', '-')}_unattended.jpg"), annotated_frame)
    return annotated_frame, detections, labels, model

    

def main(model, output_folder):
    cap = None 
    try:
        # Using webcam (device index 0)
        cap = cv2.VideoCapture("rtsp://service:Admin2024!@192.168.1.100/stream1")

        if not cap.isOpened():
            print("Error: Could not open webcam.")
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

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            print("Processing frame...")
            annotated_frame, detections, labels, model = process_frame(frame, model, box_annotator, unattended_objects, output_folder)

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

                    if class_id in CLASSES_TO_DETECT:
                        unattended = check_unattended_objects(unattended_objects, class_id, timestamp)
                        if not unattended:
                            unattended_objects[class_id].append({
                                'last_seen': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            })

            # Remove unattended objects if person is detected
            if 'person' in [d["class_name"] for d in data["detections"]]:
                for class_id in CLASSES_TO_DETECT:
                    unattended_objects[class_id] = [obj for obj in unattended_objects[class_id] if (datetime.now() - obj['last_seen']).seconds < 10]

            priority = get_priority(unattended_objects)
            if priority:
                print(f"Priority: {priority}")

            if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
                break

        # Save output data to JSON file
        with open(os.path.join(output_folder, 'muroutput.json'), 'w') as json_file:
            json.dump(output_data, json_file, default=default_serializer)

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO('yolov9e.pt')
    output_folder = 'output_frames'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    main(model, output_folder)