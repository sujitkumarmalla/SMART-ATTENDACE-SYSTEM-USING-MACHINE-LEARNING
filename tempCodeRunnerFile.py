
# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     try:
#         speaker = Dispatch("SAPI.SpVoice")
#         speaker.Speak(str1)
#     except Exception as e:
#         print("Speech Error:", e)


# os.makedirs('data', exist_ok=True)
# os.makedirs('Attendance', exist_ok=True)


# video = cv2.VideoCapture(0)
# if not video.isOpened():
#     print("Error: Webcam not found.")
#     exit()


# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# if facedetect.empty():
#     print("Error: Haarcascade not loaded.")
#     exit()


# try:
#     with open('data/names.pkl', 'rb') as f:
#         LABELS = pickle.load(f)
#     with open('data/faces_data.pkl', 'rb') as f:
#         FACES = pickle.load(f)


#     LABELS = np.array(LABELS).astype(str)  
#     FACES = np.array(FACES)

  
#     if len(FACES) != len(LABELS):
#         raise ValueError(f"Data mismatch: {len(FACES)} faces but {len(LABELS)} labels")

# except Exception as e:
#     print("Training data missing or invalid:", e)
#     exit()

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)


# imgBackground = cv2.imread("background.jpg")
# if imgBackground is None:
#     print("Background not found.")
#     exit()


# box_x, box_y = 50, 110
# box_width, box_height = 635, 385

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Webcam read failed.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     current_attendance = None

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

#         # KNN distance check
#         distances, _ = knn.kneighbors(resized_img)
#         threshold = 2000
#         if distances[0][0] > threshold:
#             name = "Unknown"
#         else:
#             name = knn.predict(resized_img)[0]

#         timestamp = datetime.now().strftime('%H:%M:%S')
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#         if name != "Unknown":
#             current_attendance = [name, timestamp]

#     # Embed frame in background
#     resized_frame = cv2.resize(frame, (box_width, box_height))
#     frame_to_show = imgBackground.copy()
#     frame_to_show[box_y:box_y + box_height, box_x:box_x + box_width] = resized_frame

#     cv2.imshow("Face Recognition & Attendance", frame_to_show)

#     key = cv2.waitKey(1)

#     if key == ord('o'):
#         if current_attendance:
#             speak("Attendance taken successfully. Thank you")
#             date = datetime.now().strftime('%d-%m-%Y')
#             filename = f"Attendance/Attendance_{date}.csv"
#             file_exists = os.path.isfile(filename)
#             with open(filename, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 if not file_exists:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow(current_attendance)
#             print(f"✅ Marked: {current_attendance}")
#         else:
#             speak("Unknown person. Please try again")
#             print("Unknown person, attendance not recorded.")

#     if key == ord('q'):
#         print("👋 Exiting...")
#         break

# video.release()
# cv2.destroyAllWindows()








# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# from datetime import datetime
# from win32com.client import Dispatch
# import pandas as pd

# def speak(str1):
#     try:
#         speaker = Dispatch("SAPI.SpVoice")
#         speaker.Speak(str1)
#     except Exception as e:
#         print("Speech Error:", e)

# os.makedirs('data', exist_ok=True)
# os.makedirs('Attendance', exist_ok=True)

# video = cv2.VideoCapture(0)
# if not video.isOpened():
#     print("Error: Webcam not found.")
#     exit()

# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# if facedetect.empty():
#     print("Error: Haarcascade not loaded.")
#     exit()

# try:
#     with open('data/names.pkl', 'rb') as f:
#         LABELS = pickle.load(f)
#     with open('data/faces_data.pkl', 'rb') as f:
#         FACES = pickle.load(f)

#     LABELS = np.array(LABELS).astype(str)  
#     FACES = np.array(FACES)

#     if len(FACES) != len(LABELS):
#         raise ValueError(f"Data mismatch: {len(FACES)} faces but {len(LABELS)} labels")

# except Exception as e:
#     print("Training data missing or invalid:", e)
#     exit()

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground = cv2.imread("background.jpg")
# if imgBackground is None:
#     print("Background not found.")
#     exit()

# box_x, box_y = 50, 110
# box_width, box_height = 635, 385
# COL_NAMES = ['NAME', 'ENTRY', 'EXIT', 'WORK_HOURS']

# def mark_attendance(name):
#     date = datetime.now().strftime('%d-%m-%Y')
#     filename = f"Attendance/Attendance_{date}.csv"

#     if not os.path.exists(filename):
#         df = pd.DataFrame(columns=COL_NAMES)
#         df.to_csv(filename, index=False)

#     df = pd.read_csv(filename)

#     if name in df['NAME'].values:
#         idx = df[df['NAME'] == name].index[0]
#         if pd.isna(df.loc[idx, 'EXIT']):  # If exit not marked yet
#             exit_time = datetime.now()
#             df.loc[idx, 'EXIT'] = exit_time.strftime('%H:%M:%S')

#             entry_time = datetime.strptime(df.loc[idx, 'ENTRY'], '%H:%M:%S')
#             work_hours = (exit_time - entry_time).seconds / 3600
#             df.loc[idx, 'WORK_HOURS'] = round(work_hours, 2)
#             speak(f"Exit marked for {name}")
#         else:
#             speak(f"{name}, you have already marked exit today.")
#     else:
#         entry_time = datetime.now().strftime('%H:%M:%S')
#         df = pd.concat([df, pd.DataFrame([[name, entry_time, None, 0]], columns=COL_NAMES)], ignore_index=True)
#         speak(f"Entry marked for {name}")

#     df.to_csv(filename, index=False)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Webcam read failed.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     current_name = None

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

#         distances, _ = knn.kneighbors(resized_img)
#         threshold = 2000
#         if distances[0][0] > threshold:
#             name = "Unknown"
#         else:
#             name = knn.predict(resized_img)[0]

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#         if name != "Unknown":
#             current_name = name

#     resized_frame = cv2.resize(frame, (box_width, box_height))
#     frame_to_show = imgBackground.copy()
#     frame_to_show[box_y:box_y + box_height, box_x:box_x + box_width] = resized_frame

#     cv2.imshow("Face Recognition & Attendance", frame_to_show)

#     key = cv2.waitKey(1)
#     if key == ord('o') and current_name:
#         mark_attendance(current_name)
#     if key == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()




from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from win32com.client import Dispatch
import time

# ---------------- CONFIG ----------------
DATA_DIR = 'data'
ATTEND_DIR = 'Attendance'
BACKGROUND_IMG = 'background.jpg'        # optional: if missing script will show raw webcam feed
FACE_SIZE = (50, 50)
KNN_NEIGHBORS = 5
DISTANCE_THRESHOLD = 2000
COOLDOWN_SECONDS = 3                     # avoid multiple immediate writes for same person
SPLIT_HOUR = 12                          # < SPLIT_HOUR => ENTRY period; >= SPLIT_HOUR => EXIT period
# If you meant midnight (12:00 AM) use: SPLIT_HOUR = 0
COL_NAMES = ['NAME', 'ENTRY', 'EXIT', 'WORK_HOURS']
# ----------------------------------------

# text-to-speech helper (Windows)
def speak(txt):
    try:
        Dispatch("SAPI.SpVoice").Speak(txt)
    except Exception as e:
        print("TTS error:", e)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTEND_DIR, exist_ok=True)

# Try to load Haarcascade using OpenCV's data path (robust)
haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ------------------ LOAD TRAINING DATA ------------------
names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

if not os.path.exists(names_path) or not os.path.exists(faces_path):
    print("Training data missing. Ensure 'data/names.pkl' and 'data/faces_data.pkl' exist.")
    exit()

with open(names_path, 'rb') as f:
    LABELS = np.array(pickle.load(f)).astype(str)

with open(faces_path, 'rb') as f:
    FACES = np.array(pickle.load(f))

if len(FACES) != len(LABELS):
    print(f"Data mismatch: {len(FACES)} faces vs {len(LABELS)} labels. Fix the data first.")
    exit()

knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
knn.fit(FACES, LABELS)

# ------------------ VIDEO + CASCADE ---------------------
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Webcam not found.")
    exit()

face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    print("Error: Haarcascade could not be loaded from:", haar_path)
    exit()

# Load optional background
imgBackground = cv2.imread(BACKGROUND_IMG)
USE_BACKGROUND = imgBackground is not None
if USE_BACKGROUND:
    # bounding box coordinates same as your previous code:
    BOX_X, BOX_Y, BOX_W, BOX_H = 50, 110, 635, 385
else:
    # if no background, we'll display raw frame
    BOX_X = BOX_Y = BOX_W = BOX_H = None

# cooldown map to avoid repeated writes
_last_mark_time = {}

# --------- helper: parse time string safely ----------
def parse_time_str(timestr):
    if pd.isna(timestr):
        return None
    if not isinstance(timestr, str):
        return None
    timestr = timestr.strip()
    if not timestr:
        return None
    try:
        return datetime.strptime(timestr, '%H:%M:%S')
    except Exception:
        return None

# --------- ensure attendance file & convert old 'TIME' format if present ----------
def ensure_attendance_file_for_date(date_str):
    """
    Ensures Attendance/Attendance_{date_str}.csv exists and has columns NAME,ENTRY,EXIT,WORK_HOURS.
    If old format with 'TIME' exists it will convert rows into the new format respecting SPLIT_HOUR rule:
      - earliest TIME before SPLIT_HOUR -> ENTRY
      - latest TIME at/after SPLIT_HOUR -> EXIT
    """
    filename = os.path.join(ATTEND_DIR, f"Attendance_{date_str}.csv")
    if not os.path.exists(filename):
        pd.DataFrame(columns=COL_NAMES).to_csv(filename, index=False)
        return filename

    df = pd.read_csv(filename)

    # If old style file has columns ['NAME','TIME'] or contains 'TIME', convert
    if 'TIME' in df.columns and (('ENTRY' not in df.columns) or ('EXIT' not in df.columns)):
        new_rows = []
        grouped = df.groupby('NAME')['TIME'].apply(list).to_dict()
        for name, times in grouped.items():
            entry = None
            exit_ = None
            parsed = []
            for t in times:
                try:
                    parsed_dt = datetime.strptime(str(t).strip(), '%H:%M:%S')
                    parsed.append(parsed_dt)
                except Exception:
                    pass
            if parsed:
                # earliest before split
                before_split = [p for p in parsed if p.hour < SPLIT_HOUR]
                after_split  = [p for p in parsed if p.hour >= SPLIT_HOUR]
                if before_split:
                    entry = min(before_split).strftime('%H:%M:%S')
                if after_split:
                    exit_ = max(after_split).strftime('%H:%M:%S')
            # compute work_hours if both present
            work_hours = 0
            if entry and exit_:
                e_dt = datetime.strptime(entry, '%H:%M:%S')
                x_dt = datetime.strptime(exit_, '%H:%M:%S')
                if x_dt < e_dt:
                    x_dt += timedelta(days=1)
                work_hours = round((x_dt - e_dt).seconds / 3600, 2)
            new_rows.append([name, entry, exit_, work_hours])
        new_df = pd.DataFrame(new_rows, columns=COL_NAMES)
        new_df.to_csv(filename, index=False)
        return filename

    # Otherwise, add any missing columns
    for col in COL_NAMES:
        if col not in df.columns:
            if col == 'WORK_HOURS':
                df[col] = 0
            else:
                df[col] = None
    # reorder columns
    df = df[COL_NAMES]
    df.to_csv(filename, index=False)
    return filename

# ------------- MARK ATTENDANCE LOGIC --------------------
def mark_attendance(name):
    """
    Behavior:
      - If current hour < SPLIT_HOUR: this is ENTRY period.
          * If student already has ENTRY -> keep earliest (do not overwrite).
          * If no ENTRY -> set ENTRY to current time.
      - If current hour >= SPLIT_HOUR: this is EXIT period.
          * Always update EXIT to the latest time.
          * If ENTRY exists -> compute WORK_HOURS = exit - entry (if exit < entry assume next day).
          * If ENTRY missing -> WORK_HOURS set to 0.
    """
    now = datetime.now()
    date_str = now.strftime('%d-%m-%Y')
    filename = ensure_attendance_file_for_date(date_str)

    # cooldown per person
    last = _last_mark_time.get(name)
    if last:
        if (now - last).total_seconds() < COOLDOWN_SECONDS:
            print(f"[{name}] action ignored (cooldown).")
            return
    # update last mark time immediately (so multi-presses don't spam)
    _last_mark_time[name] = now

    df = pd.read_csv(filename)

    # ensure columns are present (paranoid)
    for col in COL_NAMES:
        if col not in df.columns:
            df[col] = None

    if name in df['NAME'].values:
        idx = df[df['NAME'] == name].index[0]
        entry_val = df.at[idx, 'ENTRY']
        exit_val  = df.at[idx, 'EXIT']

        if now.hour < SPLIT_HOUR:
            # ENTRY period: keep earliest entry
            if pd.isna(entry_val) or str(entry_val).strip() == '':
                df.at[idx, 'ENTRY'] = now.strftime('%H:%M:%S')
                print(f"[{name}] Entry recorded at {df.at[idx,'ENTRY']}")
                speak(f"Entry marked for {name}")
            else:
                print(f"[{name}] Entry already exists: {entry_val} (not overwritten)")
                speak(f"Entry already recorded for {name}")
        else:
            # EXIT period: always update exit to latest
            df.at[idx, 'EXIT'] = now.strftime('%H:%M:%S')
            entry_parsed = parse_time_str(df.at[idx, 'ENTRY'])
            if entry_parsed:
                exit_parsed = parse_time_str(df.at[idx, 'EXIT'])
                if exit_parsed and exit_parsed < entry_parsed:
                    # exit next day
                    exit_parsed = exit_parsed + timedelta(days=1)
                hours = round((exit_parsed - entry_parsed).seconds / 3600, 2) if entry_parsed else 0
                df.at[idx, 'WORK_HOURS'] = hours
                print(f"[{name}] Exit updated to {df.at[idx,'EXIT']} | Work hours: {hours}")
            else:
                # no entry: work hours zero
                df.at[idx, 'WORK_HOURS'] = 0
                print(f"[{name}] Exit updated to {df.at[idx,'EXIT']}. No entry found -> Work hours set to 0")
            speak(f"Exit updated for {name}")
    else:
        # new person for the day
        if now.hour < SPLIT_HOUR:
            new_row = [name, now.strftime('%H:%M:%S'), None, 0]
            print(f"[{name}] New entry created at {new_row[1]}")
            speak(f"Entry marked for {name}")
        else:
            new_row = [name, None, now.strftime('%H:%M:%S'), 0]
            print(f"[{name}] New row created with exit at {new_row[2]} (no entry)")
            speak(f"Exit marked for {name}")
        new_df = pd.DataFrame([new_row], columns=COL_NAMES)
        df = pd.concat([df, new_df], ignore_index=True)

    # Save
    df.to_csv(filename, index=False)


# --------------- MAIN LOOP ------------------
print("Press 'o' to mark attendance for the currently detected person.")
print("Press 'q' to quit.")
current_name = None

while True:
    ret, frame = video.read()
    if not ret:
        print("Camera read failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_name = None
    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        resized = cv2.resize(crop, FACE_SIZE).flatten().reshape(1, -1)

        distances, _ = knn.kneighbors(resized)
        if distances[0][0] > DISTANCE_THRESHOLD:
            name = "Unknown"
        else:
            name = knn.predict(resized)[0]

        # Draw and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if name != "Unknown":
            current_name = name

    # Show either background+frame or just frame
    if USE_BACKGROUND:
        try:
            small = cv2.resize(frame, (BOX_W, BOX_H))
            out = imgBackground.copy()
            out[BOX_Y:BOX_Y+BOX_H, BOX_X:BOX_X+BOX_W] = small
            cv2.imshow("Face Recognition & Attendance", out)
        except Exception:
            # fallback to raw frame if any sizing issue
            cv2.imshow("Face Recognition & Attendance", frame)
    else:
        cv2.imshow("Face Recognition & Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        if current_name:
            mark_attendance(current_name)
        else:
            print("No recognized person in frame to mark.")
            speak("please try again.")
    elif key == ord('q'):
        print("Exiting.")
        break

video.release()
cv2.destroyAllWindows()
