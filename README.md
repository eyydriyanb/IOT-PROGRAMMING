# IOT-PROGRAMMING

# Face Recognition with IP Camera

This Python project uses OpenCV to perform real-time face recognition through an IP camera stream. It detects and recognizes known faces from a local dataset and logs recognized entries to a CSV file with timestamps.

## 📦 Features

- Real-time face detection using Haar cascades.
- Face recognition using LBPH (Local Binary Patterns Histograms).
- IP camera streaming (tested with mobile IP webcam).
- Logging of recognized individuals to a CSV file.
- Configurable face dataset directory.

## 🧰 Requirements

- Python 3.6+
- OpenCV (with `opencv-contrib-python`)
- NumPy

### Install Dependencies

```bash
pip install opencv-contrib-python numpy

project/
│
├── face1.py               # Main script
├── face_log.csv           # CSV log of recognized faces
└── known_faces/           # Directory of known individuals
    └── Person1/
        ├── image1.jpg
        └── image2.jpg
    └── Person2/
        ├── image1.jpg
        └── image2.jpg

🔧 Configuration
Update the following variables in face1.py as needed:

python
Copy
Edit
KNOWN_FACES_DIR = r'C:\path\to\known_faces'
CSV_LOG_FILE = 'face_log.csv'
IP_CAM_URL = 'http://<your_ip>:8080/video'
🚀 Usage
Place grayscale or color face images in subdirectories under known_faces/, one subdirectory per person.

Ensure the IP camera stream is accessible (e.g., use an Android IP Webcam app).

Run the script:

bash
Copy
Edit
python face1.py
Press q to quit the live stream.

📝 Logs
Recognized faces are logged once per session to face_log.csv with the following format:

pgsql
Copy
Edit
Name, Date, Time
⚠️ Notes
Ensure your OpenCV version includes the cv2.face module (opencv-contrib-python is required).

Face images are resized to 200x200 for training and prediction.
