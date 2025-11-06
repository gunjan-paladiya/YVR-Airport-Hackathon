import tkinter as tk
from threading import Thread, Lock
import cv2
from PIL import Image, ImageTk
import argparse
import json
import supervision as sv
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import queue
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import cv2
from PIL import Image, ImageTk
import argparse
import json
import supervision as sv
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import queue
import pandas as pd

camera_index = 0
cameras = []
output_logs = {0: '', 1: ''}
output_data = []
# Global queue for data transfer from threads to main GUI
data_queue = queue.Queue()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Video Feed")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    return args

def read_config(file_path: str):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def set_up_camera_feed():
    print("jello2")

    args = parse_arguments()
    config = read_config(args.config)
    cameras = config['cameras']
    captures = []
    for camera in cameras:
        #rtsp_url = camera['rtsp_url']
        rtsp_url = 'rtsp://service:Admin2024!@192.168.1.100/stream1'
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Error: Could not open {camera['name']}.")
        captures.append((camera, cap))
    return captures


def update_frame(capture, label):
    camera, cap = capture
    model = YOLO('yolov9c.pt')
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    while True:
        ret, frame = cap.read()
        if ret:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            labels = [f"{model.model.names[class_id]} {confidence:0.8f}"
                      for class_id, confidence in zip(detections.class_id, detections.confidence)]
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
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
           
            
           
            print("\n\n======================================")
            print(cumulative_counts)

           
           
            data_queue.put(cumulative_counts)  # Put data into queue for main thread to process
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            label.config(image=frame)
            label.image = frame
        else:
            break
    cap.release()

    
def main():
    root = tk.Tk()
    root.title("Multiple Camera Feeds")
    root.geometry("1280x960")

    # Create frames for video feeds and text display
    video_frame = tk.Frame(root)
    video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    text_frame = tk.Frame(root)
    text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Create a label in the text frame for displaying cumulative counts of 'person'
    counts_label = tk.Label(text_frame, text="Waiting for 'person' counts...", font=('Helvetica', 12))
    counts_label.pack(expand=True)

    captures = set_up_camera_feed()

    video_labels = []
    if captures:
        for capture in captures:
            camera, _ = capture
            video_label = tk.Label(video_frame)
            video_label.pack(expand=True, fill=tk.BOTH)
            video_labels.append(video_label)
            thread = Thread(target=update_frame, args=(capture, video_label))
            thread.daemon = True
            thread.start()

    def check_queue():
        try:
            cumulative_counts = data_queue.get_nowait()
            counts_text = ""
            show_popup = False
            for timestamp, counts in cumulative_counts.items():
                if 'person' in counts:                  
                    person_count = counts['person']
                    counts_text += f"Timestamp: {timestamp} -> Person: {person_count}\n"
                    if person_count > 2:
                        show_popup = True
                else:
                    counts_text += f"Timestamp: {timestamp} -> Person: 0\n"
            counts_label.config(text=counts_text)
            if show_popup:
                messagebox.showinfo("Alert", "More than 2 persons detected!")
        except queue.Empty:
            pass
        root.after(100, check_queue)

    root.after(100, check_queue)
    root.mainloop()

if __name__ == "__main__":
    main()
