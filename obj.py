import cv2
import numpy as np
import winsound



# Paths to YOLO model files
cfg_path = 'yolov3.cfg'  # Ensure this is the path to your .cfg file
weights_path = 'yolov3.weights'  # Ensure this is the path to your .weights file
names_path = 'coco.names'  # Ensure this is the path to your .names file

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()

# Fix: Adjust for scalar or list output
unconnected_out_layers = net.getUnconnectedOutLayers()
if isinstance(unconnected_out_layers, np.ndarray):
    unconnected_out_layers = unconnected_out_layers.flatten()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture for laptop camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

# Set camera resolution (width x height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Get screen size
screen_width = 1900 # Update this based on your screen resolution
screen_height = 1080  # Update this based on your screen resolution

# Set window size to 3/4th of the screen
window_width = int(screen_width * 0.75)
window_height = int(screen_height * 0.75)

# Create a named window with the desired size
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for the model
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    height, width, _ = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            play_sound(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
            display_fact(frame, label, x, y)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
