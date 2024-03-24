# Object Detection and Clustering Pipeline

This Flask application performs object detection using YOLOv4-tiny model and clusters the detected objects based on colour similarity using the KMeans algorithm.

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

**Note:** Sample images are provided in the `sample_images/` directory. You can test the application using these images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
