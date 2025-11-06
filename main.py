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
    
    output_data = {}
    #class_counts = {} # Needed if we are printing the output in csv
    terminal_output = []

    for idx, camera in enumerate(config['cameras']):
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

        cv2.namedWindow(camera['name'], cv2.WINDOW_NORMAL)

        output_data[camera['name']] = []
        #class_counts[camera['name']] = {} # Needed if we are printing the output in csv

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
                    "timestamp": timestamp,
                    "detections": [{
                        "class_name": model.model.names[class_id],
                        "confidence": float(confidence)
                    } for class_id, confidence in zip(detections.class_id, detections.confidence)]
                }

                output_data[camera['name']].append(data)

                # # Update class counts
                # unique_detections = set()
                # for detection in data["detections"]:
                #     timestamp = data["timestamp"]
                #     class_name = detection["class_name"]
                    
                #     if class_name not in unique_detections:
                #         unique_detections.add(class_name)
                        
                #         if timestamp not in class_counts[camera['name']]:
                #             class_counts[camera['name']][timestamp] = {}
                        
                #         class_counts[camera['name']][timestamp][class_name] = class_counts[camera['name']][timestamp].get(class_name, 0) + 1

                if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
                    break

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, saving output data...")
            
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

    # Save output data to JSON files for each camera
    for camera_name, data_list in output_data.items():
        try:
            with open(os.path.join(output_dir, f"output_{camera_name.replace(' ', '_')}.json"), "w") as f:
                json.dump(data_list, f, indent=4)
        except Exception as e:
            print(f"Error occurred while saving output data for {camera_name}: {e}")

    # # Save class counts to CSV files for each camera
    # for camera_name, counts in class_counts.items():
    #     try:
    #         with open(os.path.join(output_dir, f"class_counts_{camera_name.replace(' ', '_')}.csv"), "w", newline="") as csvfile:
    #             fieldnames = ["Timestamp", "Class Name", "Count"]
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #             writer.writeheader()
    #             for timestamp, class_data in counts.items():
    #                 for class_name, count in class_data.items():
    #                     writer.writerow({"Timestamp": timestamp, "Class Name": class_name, "Count": count})
    #     except Exception as e:
    #         print(f"Error occurred while saving class counts for {camera_name}: {e}")

if __name__ == "__main__":
    main()
