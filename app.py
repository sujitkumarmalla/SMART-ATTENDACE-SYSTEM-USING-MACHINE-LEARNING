# import streamlit as st
# import pandas as pd
# import time
# from datetime import datetime
# import os


# ts = time.time()
# date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
# timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

            
# attendance_dir = "Attendance"

# attendance_file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")


# if not os.path.exists(attendance_dir):
#     os.makedirs(attendance_dir)
#     st.info(f"Created directory: {attendance_dir}")


# if not os.path.exists(attendance_file_path):
#     df = pd.DataFrame(columns=['Name', 'Time'])
#     df.to_csv(attendance_file_path, index=False)
#     st.info(f"Created new attendance file for today: {attendance_file_path}")
# else:
    
#     df = pd.read_csv(attendance_file_path)


# st.title("Attendance Dashboard")

# from streamlit_autorefresh import st_autorefresh


# count = st_autorefresh(interval=2000, key="fizzbuzzcounter")

# if count == 0:
#     st.write("Refreshing...")

# st.subheader("Today's Attendance")
# st.dataframe(df.style.highlight_max(axis=0))


import cv2
import pickle
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from streamlit_autorefresh import st_autorefresh

# ===============================
# Load saved face data & names
# ===============================
faces_path = "data/faces_data.pkl"
names_path = "data/names.pkl"

if not (os.path.exists(faces_path) and os.path.exists(names_path)):
    st.error("No trained face data found! Please run the face collection script first.")
    st.stop()

with open(faces_path, 'rb') as f:
    faces = pickle.load(f)

with open(names_path, 'rb') as f:
    names = pickle.load(f)

# ===============================
# Train KNN model
# ===============================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, names)

# ===============================
# Attendance file setup
# ===============================
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
attendance_dir = "Attendance"
attendance_file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")

if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

if not os.path.exists(attendance_file_path):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(attendance_file_path, index=False)

# ===============================
# Face detection
# ===============================
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def mark_attendance(name):
    df = pd.read_csv(attendance_file_path)

    # If person already marked, skip
    if name != "Unknown" and name not in df['Name'].values:
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        df.loc[len(df)] = [name, timestamp]
        df.to_csv(attendance_file_path, index=False)

# ===============================
# Streamlit UI
# ===============================
st.title("Face Recognition Attendance System")
count = st_autorefresh(interval=2000, key="refresh")

start_btn = st.button("Start Recognition")

if start_btn:
    video = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Error accessing webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            output = knn.predict(resized_img)
            neighbor_distances, _ = knn.kneighbors(resized_img)

            # Distance threshold for unknown detection
            threshold = 2000  # adjust if needed
            if neighbor_distances.mean() > threshold:
                name = "Unknown"
            else:
                name = output[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

            mark_attendance(name)

        # Convert BGR to RGB for Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# ===============================
# Show Attendance Table
# ===============================
st.subheader("Today's Attendance")
df = pd.read_csv(attendance_file_path)
st.dataframe(df)
