import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model (Ensure you have a pre-trained model or specify your own)
model = YOLO("yolov8n.pt")  # Replace with the model file if needed, like yolov8n.pt

# Open the webcam (or specify a video stream)
cap = cv2.VideoCapture(0)  # 0 for default webcam or provide your video path

# Check if the video capture is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Load graphics for overlay (ensure it's the correct format)
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

# Initialize counters
people_counter = 0
previous_frame_people = []

while True:
    # Read each frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Use YOLO model to detect objects in the frame
    results = model(frame)

    # Get people detections (filter only person detections)
    people_detections = []
    for result in results:
        for box in result.boxes.data:
            cls = int(box[5])  # Class ID
            if cls == 0:  # Class 0 is 'person' in YOLO
                x1, y1, x2, y2 = map(int, box[:4])
                people_detections.append((x1, y1, x2, y2))
                
    # Count the number of detected people
    people_counter = len(people_detections)
    
    # Optionally, store or print the count for display
    print(f"People Count: {people_counter}")

    # Overlay the graphic image if it fits
    max_width, max_height = frame.shape[1], frame.shape[0]
    if imgGraphics.shape[1] > max_width or imgGraphics.shape[0] > max_height:
        imgGraphics = cv2.resize(imgGraphics, (max_width, max_height))

    hf, wf = imgGraphics.shape[:2]
    overlay_position = (min(730, max_width - wf), min(260, max_height - hf))

    # Add overlay image (make sure it has an alpha channel for transparency)
    if imgGraphics.shape[2] == 4:  # If the image has an alpha channel (transparency)
        overlay_img = imgGraphics[:, :, :3]  # RGB channels
        alpha_channel = imgGraphics[:, :, 3:]  # Alpha channel
        
        # Get the region of interest (ROI) where the overlay will go
        roi = frame[overlay_position[1]:overlay_position[1] + hf, overlay_position[0]:overlay_position[0] + wf]
        
        # Blend the images using alpha channel
        img_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(alpha_channel))
        img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=alpha_channel)
        blended = cv2.add(img_bg, img_fg)
        
        # Place the blended image back into the frame
        frame[overlay_position[1]:overlay_position[1] + hf, overlay_position[0]:overlay_position[0] + wf] = blended
    else:
        # If no transparency, just overlay the image directly
        frame[overlay_position[1]:overlay_position[1] + hf, overlay_position[0]:overlay_position[0] + wf] = imgGraphics

    # Draw bounding boxes for each detected person
    for (x1, y1, x2, y2) in people_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result with people count
    cv2.putText(frame, f"People Count: {people_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with overlay and people count
    cv2.imshow("Live Feed", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
