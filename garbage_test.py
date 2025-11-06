import cv2
import torch

# Load the pre-trained model
model = torch.hub.load(r'/Users/murtaza_vora/yolov9', 'custom', path='best.pt', source='local')  # replace with your local repo path

# Open the video
cap = cv2.VideoCapture(r"WhatsApp Video 2024-04-12 at 02.36.53.mp4")  # replace with your video file path
  
while cap.isOpened():
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Make detections
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()