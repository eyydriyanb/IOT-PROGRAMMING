import cv2
import os
import numpy as np
from datetime import datetime
import csv

# === CONFIGURATION ===
KNOWN_FACES_DIR = r'C:\Users\Adrian Balmes\OneDrive\Desktop\masters\second sem 24-25\iot\FINALS\known_faces'
CSV_LOG_FILE = 'face_log.csv'
IP_CAM_URL = 'http://18.18.18.227:8080/video'  # <-- Replace with your iPhone IP stream

# === LOAD FACE DETECTOR ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Failed to load Haar cascade")
    exit(1)

# === INIT FACE RECOGNIZER ===
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("❌ OpenCV missing 'cv2.face'. Install with:")
    print("    pip install opencv-contrib-python")
    exit(1)

# === LOAD KNOWN FACES ===
faces, labels, label_names = [], [], {}
current_label = 0
print("📂 Loading known faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir): continue

    label_names[current_label] = person_name

    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue
            img_resized = cv2.resize(img, (200, 200))
            faces.append(img_resized)
            labels.append(current_label)

    current_label += 1

if not faces:
    print("❌ No training images found.")
    exit(1)

recognizer.train(faces, np.array(labels))
print(f"✅ Training complete on {len(faces)} images.")

# === OPEN IP CAMERA STREAM ===
cap = cv2.VideoCapture(IP_CAM_URL)
if not cap.isOpened():
    print("❌ Cannot open IP camera")
    exit(1)

print("🎥 Starting IP webcam... Press 'q' to quit.")

# === LOGGING HELPERS ===
logged_names = set()

def log_to_csv(name):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    timestamp = now.strftime('%H:%M:%S')

    with open(CSV_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date, timestamp])
        print(f"📝 Logged: {name}, {date}, {timestamp}")

# === VIDEO LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame error. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_resized)
        name = label_names.get(label, "Unknown")

        if confidence < 85:
            color = (0, 255, 0)
            text = f"{name} ({int(confidence)})"

            # Log once per session per person
            if name not in logged_names:
                log_to_csv(name)
                logged_names.add(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            text = name

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - IP Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
