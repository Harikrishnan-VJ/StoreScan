# StoreScan: Object Detection and Clustering Pipeline

StoreScan is a Flask application designed to facilitate product categorization in images. It utilizes advanced object detection and clustering techniques to identify products within images and group them based on color similarity. This application provides an intuitive web interface for users to upload images and receive JSON-formatted detection results.

## Project Overview

StoreScan simplifies the process of analyzing retail shelf images by automating the detection and categorization of products. Users can upload images containing various products, and the application will detect each product and assign it to a specific category based on color similarity. This streamlines inventory management and product organization tasks for retail businesses.

## Application Design

StoreScan serves as an AI pipeline with the following key components:

1. **Flask Webserver:** Handles HTTP requests and provides a user-friendly web interface for uploading images.
2. **Detection Model:** Utilizes the YOLOv4-tiny model trained on the SKU110K Dataset to detect products within images accurately.
3. **Product Grouping:** Employs the KMeans algorithm to cluster detected products based on color features, allowing for efficient categorization.

This application streamlines the process of product detection and categorization, enhancing productivity and accuracy in retail inventory management.

## Dataset

The dataset used for training the object detection model is the SKU110K Dataset, available at [Kaggle](https://www.kaggle.com/datasets/thedatasith/sku110k-annotations). This dataset provides annotated images of products commonly found on retail shelves, making it ideal for training object detection models for retail applications.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/Harikrishnan-VJ/StoreScan.git
    cd StoreScan
    ```

2. Install dependencies:
    ```
    pip3 install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```
    python3 app.py
    ```

2. Open a web browser and go to `http://127.0.0.1:5000/` to access the web interface.

3. **App UI Steps:**
    - Click on "Choose Image" and select the image you want to analyze.
    - Click "Upload" and wait for the output to be displayed.
    - Once the output is shown, click on the "Download JSON" button at the bottom of the image to download the JSON file containing the detected objects.

## Files and Directories

- `app.py`: Flask application file containing routes and main functionality.
- `index.html`: HTML template for the home page.
- `obj.names`: File containing class names detected by YOLOv4-tiny.
- `yolov4-tiny.cfg`: YOLOv4-tiny configuration file.
- `yolov4-tiny.weights`: YOLOv4-tiny pre-trained weights file.
- `sample_images/`: Directory containing sample images for testing the application.
- `requirements.txt`: File containing Python dependencies.

**Note:** Feel free to test the application using the sample images provided in the sample_images/ directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
