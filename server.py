from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
import os
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

# Define paths for YOLOv4-tiny configuration and weights
yolo_config_path = "/home/trois/darknet/yolov4-tiny-sk.cfg"
yolo_weights_path = "/home/trois/darknet/yolov4-tiny-sk_best.weights"
coco_names_path = "/home/trois/darknet/obj-sk.names"


# Load YOLOv4-tiny model and class names
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)


# Explicitly specify the output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

with open(coco_names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if file part exists in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    # Get file from request
    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read image file
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    detected_objects, image_with_boxes = perform_object_detection(image)

    # Encode image to base64
    _, img_encoded = cv2.imencode('.png', image_with_boxes)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Save JSON data to file
    json_filename = 'detected_objects.json'
    json_path = os.path.join(app.root_path, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(detected_objects, json_file)

    # Return JSON response with detected objects, base64 encoded image, and JSON filename
    return jsonify({'detected_objects': detected_objects, 'image_with_boxes': img_base64, 'json_filename': json_filename})

# Define route to download JSON file
@app.route('/download/<filename>')
def download_json(filename):
    return send_from_directory(app.root_path, filename)

def perform_object_detection(image):
    height, width = image.shape[:2]

    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set input blob
    net.setInput(blob)

    # Forward pass
    output_layers = net.forward(output_layer_names)

    # Process outputs
    detected_objects = []
    image_with_boxes = image.copy()
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Scale bounding box coordinates to image size
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                detected_objects.append({
                    'class_name': class_names[class_id],
                    'confidence': float(confidence),
                    'x': x,
                    'y': y,
                    'width': int(w),
                    'height': int(h)
                })

                # Draw bounding box on image
                cv2.rectangle(image_with_boxes, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)

    return detected_objects, image_with_boxes


if __name__ == '__main__':
    app.run(debug=True)
