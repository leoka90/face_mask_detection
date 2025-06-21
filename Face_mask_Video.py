import cv2
import numpy as np
import pygame
import time
from tensorflow.keras.models import load_model  # type: ignore


pygame.mixer.init()
pygame.mixer.music.load("alert (online-audio-converter.com).wav")

img_size = 100
cooldown = 3  # seconds between alerts

# Load model and face detector
model = load_model('mask_detector_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_alert_time = 2

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (img_size, img_size))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def detect_mask(frame):
    global last_alert_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_region)

        prediction = model.predict(processed_face)[0][0]
        mask_prob = prediction
        no_mask_prob = 1 - prediction

        if prediction < 0.5:
            label_text = "Mask"
            confidence = mask_prob
            color = (0, 255, 0)
            print(f"Mask Detected ({confidence * 100:.2f}%)")
        else:
            label_text = "No Mask"
            confidence = no_mask_prob
            color = (0, 0, 255)
            print(f"ALERT: No Mask Detected! ({confidence * 100:.2f}%)")

            current_time = time.time()
            if not pygame.mixer.music.get_busy() and (current_time - last_alert_time) > cooldown:
                pygame.mixer.music.play()
                last_alert_time = current_time

        label = f"{label_text}: {confidence * 100:.2f}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_mask_detection.avi', fourcc, 20.0, (frame_width, frame_height))

print("Starting Face Mask Detection on Video. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_mask(frame)
    cv2.imshow('Face Mask Detector (Video)', frame)

    out.write(frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Detection stopped and video recorded as 'output_with_mask_detection.avi'")
