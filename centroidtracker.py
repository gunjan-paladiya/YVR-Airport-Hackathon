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

def main():
    args = parse_arguments()
    config = read_config(args.config)
    
    output_data = []
    class_counts = {}
    terminal_output = []

    for camera in config['cameras']:
        rtsp_url = camera['rtsp_url']
        frame_width, frame_height = camera['frame_resolution']

        # Extracting username and password from RTSP URL
        credentials = rtsp_url.split("://")[1].split("@")[0].split(":")
        username = credentials[0]
        password = credentials[1]

        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open {camera['name']}.")
            continue

        # Load YOLO model
        model = YOLO('yolov9c.pt')

        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Could not read frame from {camera['name']}.")
                    break

                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)

                terminal_output.append(f"Detections from {camera['name']}: {detections}")

                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]

                terminal_output.append(f"Labels from {camera['name']}: {labels}")

                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels)

                cv2.imshow(camera['name'], frame)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                data = {
                    "camera": camera['name'],
                    "timestamp": timestamp,
                    "detections": [{
                        "class_name": model.model.names[class_id],
                        "confidence": float(confidence)
                    } for class_id, confidence in zip(detections.class_id, detections.confidence)]
                }

                output_data.append(data)

                # Update class counts
                unique_detections = set()
                for detection in data["detections"]:
                    timestamp = data["timestamp"]
                    class_name = detection["class_name"]
                    
                    if class_name not in unique_detections:
                        unique_detections.add(class_name)
                        
                        if timestamp not in class_counts:
                            class_counts[timestamp] = {}
                        
                        class_counts[timestamp][class_name] = class_counts[timestamp].get(class_name, 0) + 1

                if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
                    break

        except Exception as e:
            print(f"Error occurred during processing {camera['name']}: {e}")

        finally:
            cap.release()

    cv2.destroyAllWindows()

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save terminal output to a text file
    try:
        with open(os.path.join(output_dir, "terminal_output.txt"), "w") as f:
            for line in terminal_output:
                f.write(f"{line}\n")
    except Exception as e:
        print(f"Error occurred while saving terminal output: {e}")

    # Save output data to JSON file
    try:
        with open(os.path.join(output_dir, "output.json"), "w") as f:
            json.dump(output_data, f, indent=4)
    except Exception as e:
        print(f"Error occurred while saving output data: {e}")

    # Save class counts to CSV file
    try:
        with open(os.path.join(output_dir, "class_counts.csv"), "w", newline="") as csvfile:
            fieldnames = ["Camera", "Timestamp", "Class Name", "Count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in output_data:
                camera_name = data["camera"]
                timestamp = data["timestamp"]
                for detection in data["detections"]:
                    class_name = detection["class_name"]
                    count = class_counts[timestamp][class_name]
                    writer.writerow({"Camera": camera_name, "Timestamp": timestamp, "Class Name": class_name, "Count": count})
    except Exception as e:
        print(f"Error occurred while saving class counts: {e}")

if __name__ == "__main__":
    main()
