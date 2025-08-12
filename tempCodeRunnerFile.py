
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    try:
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Speak(str1)
    except Exception as e:
        print("Speech Error:", e)


os.makedirs('data', exist_ok=True)
os.makedirs('Attendance', exist_ok=True)


video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Webcam not found.")
    exit()


facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Haarcascade not loaded.")
    exit()


try:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)


    LABELS = np.array(LABELS).astype(str)  
    FACES = np.array(FACES)

  
    if len(FACES) != len(LABELS):
        raise ValueError(f"Data mismatch: {len(FACES)} faces but {len(LABELS)} labels")

except Exception as e:
    print("Training data missing or invalid:", e)
    exit()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


imgBackground = cv2.imread("background.jpg")
if imgBackground is None:
    print("Background not found.")
    exit()


box_x, box_y = 50, 110
box_width, box_height = 635, 385

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        print("Webcam read failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    current_attendance = None

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # KNN distance check
        distances, _ = knn.kneighbors(resized_img)
        threshold = 2000
        if distances[0][0] > threshold:
            name = "Unknown"
        else:
            name = knn.predict(resized_img)[0]

        timestamp = datetime.now().strftime('%H:%M:%S')
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if name != "Unknown":
            current_attendance = [name, timestamp]

    # Embed frame in background
    resized_frame = cv2.resize(frame, (box_width, box_height))
    frame_to_show = imgBackground.copy()
    frame_to_show[box_y:box_y + box_height, box_x:box_x + box_width] = resized_frame

    cv2.imshow("Face Recognition & Attendance", frame_to_show)

    key = cv2.waitKey(1)

    if key == ord('o'):
        if current_attendance:
            speak("Attendance taken successfully. Thank you")
            date = datetime.now().strftime('%d-%m-%Y')
            filename = f"Attendance/Attendance_{date}.csv"
            file_exists = os.path.isfile(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(current_attendance)
            print(f"✅ Marked: {current_attendance}")
        else:
            speak("Unknown person. Please try again")
            print("Unknown person, attendance not recorded.")

    if key == ord('q'):
        print("👋 Exiting...")
        break

video.release()
cv2.destroyAllWindows()
