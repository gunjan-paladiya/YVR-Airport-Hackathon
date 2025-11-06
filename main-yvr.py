import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime, timedelta
import os
from ultralytics import YOLO
from collections import defaultdict

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
    last_seen = max([obj['last_seen'] for objs in unattended_objects.values() for obj in objs])
    time_difference = (datetime.now() - last_seen).seconds
    if time_difference >= 30:
        return "P1"
    elif time_difference >= 20:
        return "P2"
    elif time_difference >= 10:
        return "P3"
    return None

def process_frame(frame, model, box_annotator):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.4f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels)

    return annotated_frame, detections, labels, model

# Convert datetime object to string format
def default_serializer(obj):
    if isinstance(obj, (datetime, type(None))):
        return obj.strftime('%Y-%m-%d %H:%M:%S') if obj else None
    raise TypeError("Type not serializable")

def main(model):
    try:
        args = parse_arguments()
        config = read_config(args.config)
        
        # Create output directory if it doesn't exist
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cameras = config['cameras']

        for camera in cameras:
            rtsp_url = camera['rtsp_url']
            camera_name = camera['name']

            credentials = rtsp_url.split("://")[1].split("@")[0].split(":")
            username = credentials[0]
            password = credentials[1]

            print(f"Opening {camera_name}...")
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                print(f"Error: Could not open {camera_name}.")
                continue

            print(f"Setting up annotator for {camera_name}...")
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
                    print(f"Error: Could not read frame from {camera_name}.")
                    break

                print(f"Processing frame from {camera_name}...")
                annotated_frame, detections, labels, model = process_frame(frame, model, box_annotator)

                cv2.imshow(f"yolov9 - {camera_name}", annotated_frame)

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

            cap.release()
            cv2.destroyAllWindows()

            # Save terminal output and other data for each camera in the output directory
            try:
                with open(os.path.join(output_dir, f"terminal_output_{camera_name}.txt"), "w") as f:
                    for line in terminal_output:
                        f.write(f"{line}\n")
            except Exception as e:
                print(f"Error occurred while saving terminal output for {camera_name}: {e}")

            try:
                unattended_objects_with_priority = {
                    class_name: {
                        "objects": objs,
                        "priority": get_priority({class_name: objs})
                    }
                    for class_name, objs in unattended_objects.items()
                }

                with open(os.path.join(output_dir, f"unattended_objects_{camera_name}.json"), "w") as f:
                    json.dump(unattended_objects_with_priority, f, indent=4, default=default_serializer)
            except Exception as e:
                print(f"Error occurred while saving unattended objects for {camera_name}: {e}")

            try:
                with open(os.path.join(output_dir, f"output_{camera_name}.json"), "w") as f:
                    json.dump(output_data, f, indent=4)
            except Exception as e:
                print(f"Error occurred while saving output data for {camera_name}: {e}")

            try:
                with open(os.path.join(output_dir, f"class_counts_{camera_name}.csv"), "w", newline="") as csvfile:
                    fieldnames = ["Timestamp", "Class Name", "Count"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for timestamp, counts in class_counts.items():
                        for class_name, count in counts.items():
                            writer.writerow({"Timestamp": timestamp, "Class Name": class_name, "Count": count})
            except Exception as e:
                print(f"Error occurred while saving class counts for {camera_name}: {e}")

    except Exception as e:
        print(f"Error occurred during processing: {e}")

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
    model = YOLO('yolov9c.pt')
    main(model)

