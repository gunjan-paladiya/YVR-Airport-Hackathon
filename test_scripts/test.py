import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime
import os

from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv9 live")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    return args

def read_config(file_path: str):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_centroid(coord):
    x1, y1, x2, y2 = coord
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def main():
    args = parse_arguments()
    config = read_config(args.config)
    
    camera = config['cameras'][0]
    rtsp_url = camera['rtsp_url']
    frame_width, frame_height = camera['frame_resolution']

    credentials = rtsp_url.split("://")[1].split("@")[0].split(":")
    username = credentials[0]
    password = credentials[1]

    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    model = YOLO('yolov9c.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    output_data = []
    class_counts = {}
    tracked_objects = {}

    terminal_output = []

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            frame_height, frame_width, _ = frame.shape

            centroids = []
            if hasattr(detections, 'boxes'):
                for box in detections.boxes:
                    centroids.append(calculate_centroid(box))
            elif hasattr(detections, 'coordinates'):
                for coord in detections.coordinates:
                    centroids.append(calculate_centroid(coord))
            else:
                print("Error: Unsupported detections format.")
                break

            for centroid in centroids:
                if len(tracked_objects) == 0:
                    tracked_objects[centroid] = len(tracked_objects) + 1
                else:
                    matched = False
                    for tracked_centroid in list(tracked_objects.keys()):
                        if abs(centroid[0] - tracked_centroid[0]) < 50 and abs(centroid[1] - tracked_centroid[1]) < 50:
                            tracked_objects[centroid] = tracked_objects.pop(tracked_centroid)
                            matched = True
                            break
                    if not matched:
                        tracked_objects[centroid] = len(tracked_objects) + 1

            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f} - ID: {tracked_objects[calculate_centroid(coord)]:d}"
                for class_id, confidence, coord in zip(detections.class_id, detections.confidence, centroids)
            ]

            terminal_output.append(f"Detections: {detections}")
            terminal_output.append(f"Labels: {labels}")

            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels)

            cv2.imshow("yolov9", frame)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "timestamp": timestamp,
                "detections": [{
                    "class_name": model.model.names[class_id],
                    "confidence": float(confidence),
                    "tracking_id": tracked_objects[calculate_centroid(coord)]
                } for class_id, confidence, coord in zip(detections.class_id, detections.confidence, centroids)]
            }

            output_data.append(data)

            unique_detections = set()
            for detection in data["detections"]:
                timestamp = data["timestamp"]
                class_name = detection["class_name"]
                
                if class_name not in unique_detections:
                    unique_detections.add(class_name)
                    
                    if timestamp not in class_counts:
                        class_counts[timestamp] = {}
                    
                    class_counts[timestamp][class_name] = class_counts[timestamp].get(class_name, 0) + 1

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        try:
            with open("terminal_output.txt", "w") as f:
                for line in terminal_output:
                    f.write(f"{line}\n")
        except Exception as e:
            print(f"Error occurred while saving terminal output: {e}")

        try:
            with open("output.json", "w") as f:
                json.dump(output_data, f, indent=4)
        except Exception as e:
            print(f"Error occurred while saving output data: {e}")

        try:
            with open("class_counts.csv", "w", newline="") as csvfile:
                fieldnames = ["Timestamp", "Class Name", "Count"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for timestamp, counts in class_counts.items():
                    for class_name, count in counts.items():
                        writer.writerow({"Timestamp": timestamp, "Class Name": class_name, "Count": count})
        except Exception as e:
            print(f"Error occurred while saving class counts: {e}")

if __name__ == "__main__":
    main()
