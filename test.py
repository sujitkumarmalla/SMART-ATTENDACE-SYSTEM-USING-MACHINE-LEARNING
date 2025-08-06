from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    """Speaks a given string using Windows built-in TTS."""
    try:
        speak_engine = Dispatch("SAPI.SpVoice")
        speak_engine.Speak(str1)
    except Exception as e:
        print(f"Speech synthesis error: {e}")


if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')


video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()


facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Haarcascade XML not found.")
    exit()


try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except Exception as e:
    print(f"Error loading training data: {e}")
    exit()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


imgBackground = cv2.imread("background.jpg")
if imgBackground is None:
    print("Warning: Background image not found.")
    imgBackground = None


COL_NAMES = ['NAME', 'TIME']
box_x, box_y = 40, 70      
box_width, box_height = 320, 186  


while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Couldn't read webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    current_attendance = None

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        distances, _ = knn.kneighbors(resized_img)
        threshold = 5000  

        if distances[0][0] > threshold:
            name = "Unknown"
        else:
            name = knn.predict(resized_img)[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        if name != "Unknown":
            current_attendance = [name, str(timestamp)]

    
    if imgBackground is not None:
        resized_frame = cv2.resize(frame, (box_width, box_height))
        try:
            imgBackground[box_y:box_y + box_height, box_x:box_x + box_width] = resized_frame
            cv2.imshow("Frame", imgBackground)
        except Exception as e:
            print(f"Image placement error: {e}")
            cv2.imshow("Frame", frame)
    else:
        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('o') and current_attendance:
        speak("Attendance taken successfully.")
        attendance_file = os.path.join("Attendance", f"Attendance_{date}.csv")

        file_exists = os.path.isfile(attendance_file)
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(current_attendance)

        print(f"✅ Attendance recorded for {current_attendance[0]} at {current_attendance[1]}")

    if key == ord('q'):
        print("👋 Exiting...")
        break

video.release()
cv2.destroyAllWindows()
