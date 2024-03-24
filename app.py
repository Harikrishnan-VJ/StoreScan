from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
import os
import cv2
import numpy as np
import base64
import json
from sklearn.cluster import KMeans

app = Flask(__name__)

# Define paths for YOLOv4-tiny configuration and weights
yolo_config_path = "yolov4-tiny.cfg"
yolo_weights_path = "yolov4-tiny.weights"
names_path = "obj.names"

# Load YOLOv4-tiny model and class names
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Explicitly specify the output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

def extract_color_features(image, bbox):
    x, y, w, h = bbox
    # Ensure the bounding box coordinates are within the image dimensions
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if w <= 0 or h <= 0:
        return None
    
    roi = image[y:y+h, x:x+w]  # Extract ROI from the image
    
    # Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate color histogram of the ROI
    histogram = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    
    return histogram

# Function to perform object detection
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

    # Perform clustering to group objects based on color similarity
    object_features = []
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

                # Extract color features from the bounding box
                features = extract_color_features(image, (x, y, w, h))
                object_features.append(features)

                detected_objects.append({
                    'class_name': class_names[class_id],
                    'confidence': float(confidence),
                    'x': x,
                    'y': y,
                    'width': int(w),
                    'height': int(h)
                })

    # Perform clustering to group objects based on color similarity
    cluster_labels = perform_clustering(object_features)

    # Assign colors to each cluster and draw bounding boxes
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
                 (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 165, 0),
                 (255, 192, 203), (0, 255, 0), (255, 255, 255), (0, 0, 0)]

    for obj, label in zip(detected_objects, cluster_labels):
        # Draw bounding box on image with different color
        cv2.rectangle(image_with_boxes, (obj['x'], obj['y']), (obj['x'] + obj['width'], obj['y'] + obj['height']),
                      color_map[label % len(color_map)], 5)

    return detected_objects, image_with_boxes, cluster_labels


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
    detected_objects, image_with_boxes, cluster_labels = perform_object_detection(image)

    # Organize objects by cluster
    cluster_objects = {}
    for obj, label in zip(detected_objects, cluster_labels):
        cluster_id = str(label)
        if cluster_id not in cluster_objects:
            cluster_objects[cluster_id] = []
        cluster_objects[cluster_id].append(obj)

    # Generate JSON response
    json_response = {}
    for cluster_id, objects in cluster_objects.items():
        json_response[cluster_id] = objects

    # Encode image to base64
    _, img_encoded = cv2.imencode('.png', image_with_boxes)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Save JSON data to file
    json_filename = 'detected_objects.json'
    json_path = os.path.join(app.root_path, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(json_response, json_file, indent=4)

    # Return JSON response with detected objects, base64 encoded image, and JSON filename
    return jsonify({'detected_objects': json_response, 'image_with_boxes': img_base64, 'json_filename': json_filename})


# Define route to download JSON file
@app.route('/download/<filename>')
def download_json(filename):
    return send_from_directory(app.root_path, filename, as_attachment=True)

# Function to perform clustering
def perform_clustering(features):
    # Define the number of clusters
    num_clusters = 10  # You can adjust this parameter based on your requirements

    # Initialize KMeans model
    kmeans = KMeans(n_clusters=num_clusters)

    # Fit KMeans model to the features
    kmeans.fit(features)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    return cluster_labels

if __name__ == '__main__':
    app.run(debug=True)
