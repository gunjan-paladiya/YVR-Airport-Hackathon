import cv2
import argparse
import json
from datetime import datetime
from ultralytics import YOLO

def check_unattended_objects(unattended_objects, class_id, timestamp):
    current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    class_name = model.names[class_id]
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

def process_frame(frame, model, unattended_objects):
    result = model(frame)[0]
    detections = result.xyxy[0].cpu().numpy()

    for detection in detections:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        if class_name == 'backpack':
            if not check_unattended_objects(unattended_objects, class_id, timestamp):
                # Change color based on priority
                priority = get_priority(unattended_objects)
                if priority == "P1":
                    color = (0, 0, 255)  # Red for P1
                elif priority == "P2":
                    color = (0, 255, 0)  # Green for P2
                elif priority == "P3":
                    color = (255, 255, 0)  # Yellow for P3
                else:
                    color = (255, 255, 255)  # White for other cases
                x1, y1, x2, y2 = map(int, detection[:4])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame

def main(model):
    cap = None 
    try:
        args = parse_arguments()
        config = read_config(args.config)
        
        camera = config['cameras'][0]  # Assuming only one camera for simplicity
        rtsp_url = camera['rtsp_url']

        print("Opening the camera...")
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        output_data = []
        unattended_objects = defaultdict(list)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print("Processing frame...")
            frame = process_frame(frame, model, unattended_objects)

            cv2.imshow("YOLO", frame)

            if cv2.waitKey(1) == 27:  # Wait for 'Esc' key to exit
                break

    except Exception as e:
        print(f"Error occurred during processing: {e}")

    finally:
        if cap is not None:
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
    model = YOLO('yolov9.pt')
    main(model)
