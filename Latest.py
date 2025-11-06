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
        return "p1"
    elif time_difference >= 20:
        return "P2"
    elif time_difference >= 10:
        return "P3"
    return None

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

# Convert datetime object to string format
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
            annotated_frame, detections, labels, model = process_frame(frame, model, box_annotator)

            cv2.imshow("yolov9", annotated_frame)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "timestamp": timestamp,
                "detections": [{
                    "class_name": model.model.names[class_id],
                    "confidence": float(confidence)
                } for class_id, confidence in zip(detections.class_id, detections.confidence)]
            }



            timestamps_data = defaultdict(lambda: defaultdict(int))
            cumulative_counts = defaultdict(lambda: defaultdict(int))

            timestamp = data["timestamp"]
            detections = data["detections"]
            
            for detection in detections:
                class_name = detection["class_name"]
                timestamps_data[timestamp][class_name] += 1
                cumulative_counts[timestamp][class_name] = timestamps_data[timestamp][class_name]

            for prev_timestamp in cumulative_counts:
                if prev_timestamp < timestamp:
                    for class_name in cumulative_counts[prev_timestamp]:
                        cumulative_counts[timestamp][class_name] += cumulative_counts[prev_timestamp][class_name]



            fig, ax = plt.subplots()
            plt.xticks(rotation=45, ha='right')
            line, = ax.plot([], [], marker='o')

# Update function for animation
            def update(frame):
                            timestamps = sorted(cumulative_counts.keys())
                            classes = set()
                            for ts in timestamps:
                                classes.update(cumulative_counts[ts].keys())
                            classes = sorted(classes)
                            
                            x = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
                            y = [[cumulative_counts[ts][cls] for ts in timestamps] for cls in classes]

                            ax.clear()
                            ax.plot(x, y)
                            ax.legend(classes, loc='upper left')
                            ax.set_xlabel("Timestamp")
                            ax.set_ylabel("Count")
                            ax.set_title("Cumulative Counts of Classes Over Time")

            time.sleep(10)
            # Create animation
            ani = FuncAnimation(fig, update, frames=len(data), interval=1000)

            plt.show()





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

        try:
            with open("terminal_output.txt", "w") as f:
                for line in terminal_output:
                    f.write(f"{line}\n")
        except Exception as e:
            print(f"Error occurred while saving terminal output: {e}")

        # Save unattended_objects to JSON file
        try:
            # Combine unattended_objects with priority
            unattended_objects_with_priority = {
                class_name: {
                    "objects": objs,
                    "priority": get_priority({class_name: objs})
                }
                for class_name, objs in unattended_objects.items()
            }

            with open("unattended_objects.json", "w") as f:
                json.dump(unattended_objects_with_priority, f, indent=4, default=default_serializer)
        except Exception as e:
            print(f"Error occurred while saving unattended objects: {e}")


            

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



### Creating a graph

# import json
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from datetime import datetime

# # Load data from JSON file
# with open("output.json", "r") as f:
#     output_data = json.load(f)

# # Process the data to get counts of each class at each timestamp
# class_counts_by_timestamp = defaultdict(lambda: defaultdict(int))
# for entry in output_data:
#     timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
#     for detection in entry["detections"]:
#         class_name = detection["class_name"]
#         class_counts_by_timestamp[class_name][timestamp] += 1

# # Plot the data
# plt.figure(figsize=(12, 8))
# for class_name, counts in class_counts_by_timestamp.items():
#     timestamps = sorted(counts.keys())
#     class_counts = [counts[ts] for ts in timestamps]
#     plt.plot(timestamps, class_counts, label=class_name)

# plt.xlabel("Timestamp")
# plt.ylabel("Count")
# plt.title("Count of Each Class Over Time")
# plt.legend()
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()
