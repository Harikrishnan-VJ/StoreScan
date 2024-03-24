import cv2
import numpy as np
# Load YOLOv4 Tiny model and its configuration
net = cv2.dnn.readNet("/home/trois/darknet/yolov4-tiny-sk_best.weights", "/home/trois/darknet/yolov4-tiny-sk.cfg")

# Load classes
with open("/home/trois/darknet/obj-sk.names", "r") as f:
    classes = f.read().strip().split("\n")


# Explicitly specify the output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Load image
image = cv2.imread("/home/trois/darknet/sk_images/12.jpg")
height, width = image.shape[:2]


# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set input blob
net.setInput(blob)

# Forward pass
outputs = net.forward(output_layer_names)

# Process detections
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Draw bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display result
cv2.imwrite('out.jpg', image)

