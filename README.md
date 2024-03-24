# Object Detection and Clustering Pipeline

This Flask application performs object detection using YOLOv4-tiny model and clusters the detected objects based on color similarity using KMeans algorithm.

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

3. Upload an image file using the provided form.

4. The application will perform object detection and clustering on the uploaded image, and display the result with bounding boxes drawn around the detected objects.

## Files and Directories

- `app.py`: Flask application file containing routes and main functionality.
- `index.html`: HTML template for the home page.
- `obj.names`: File containing class names detected by YOLOv4-tiny.
- `yolov4-tiny.cfg`: YOLOv4-tiny configuration file.
- `yolov4-tiny.weights`: YOLOv4-tiny pre-trained weights file.
- `requirements.txt`: File containing Python dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

