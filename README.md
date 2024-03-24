# Object Detection and Clustering Pipeline

This Flask application performs object detection using YOLOv4-tiny model and clusters the detected objects based on color similarity using KMeans algorithm.

## Installation

1. Clone the repository:
    ```
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Download the YOLOv4-tiny weights and configuration file:
    - Download `yolov4-tiny.weights` and `yolov4-tiny.cfg` files and place them in the root directory of the project.

4. Create `obj.names` file:
    - Create a file named `obj.names` and add the names of the classes detected by YOLOv4-tiny, each on a new line.

## Usage

1. Run the Flask application:
    ```
    python app.py
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

