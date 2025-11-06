import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime

from ultralytics import YOLO

import json

def load_config(filename="config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config["camera"]["rtsp_url"], config["camera"]["frame_resolution"]


# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="YOLOv9 live")
#     parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
#     args = parser.parse_args()
#     return args

def main():
    rtsp_url, frame_resolution = load_config()

    frame_width, frame_height = frame_resolution

    # Connect to the Bosch IP camera using RTSP
    cap = cv2.VideoCapture(rtsp_url)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    # Load YOLO model
    model = YOLO('yolov9c.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    output_data = []
    class_counts = {}

    while True:
        ret, frame = cap.read()

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        print("Detections:", detections)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        print("Labels:", labels)

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
                "confidence": float(confidence)
            } for class_id, confidence in zip(detections.class_id, detections.confidence)]
        }

        output_data.append(data)

        # Update class counts
        for detection in data["detections"]:
            class_name = detection["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save output data to JSON file
    with open("output.json", "w") as f:
        json.dump(output_data, f, indent=4)

    # Save class counts to CSV file
    with open("class_counts.csv", "w", newline="") as csvfile:
        fieldnames = ["Class Name", "Count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for class_name, count in class_counts.items():
            writer.writerow({"Class Name": class_name, "Count": count})

if __name__ == "__main__":
    main()
