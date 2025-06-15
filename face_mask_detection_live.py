import cv2
import numpy as np
import pygame
import time
from tensorflow.keras.models import load_model # type: ignore

pygame.mixer.init()


pygame.mixer.music.load("alert (online-audio-converter.com).wav")  


img_size = 100
cooldown = 2  


model = load_model('mask_detector_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


last_alert_time = 2

def detect_mask(frame):
    global last_alert_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (img_size, img_size))
        face = np.expand_dims(face, axis=0) / 255.0

        prediction = model.predict(face)[0][0]

        if prediction > 0.5:
            label = "Mask"
            color = (0, 225, 0)
            print("Mask detected")
        else:
            label = "No Mask"
            color = (0, 0, 225)
            print("ALERT: No Mask Detected!")

            
            current_time = time.time()
            if not pygame.mixer.music.get_busy() and (current_time - last_alert_time) > cooldown:
                pygame.mixer.music.play()
                last_alert_time = current_time

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


cap = cv2.VideoCapture(0)

print("🔍 Starting Face Mask Detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_mask(frame)
    cv2.imshow('Face Mask Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
