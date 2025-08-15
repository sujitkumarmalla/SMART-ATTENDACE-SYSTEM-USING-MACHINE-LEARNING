# import cv2
# import pickle
# import numpy as np
# import os

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# faces_data = []
# i = 0
# name = input("Enter Your Name: ")


# if not os.path.exists('data'):
#     os.makedirs('data')
#     print("Created 'data' directory.")

# print(f"Collecting 100 face samples for {name}. Please look at the camera steadily.")

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Error: Could not read frame from video source. Exiting.")
#         break
        
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50))
        
  
#         if len(faces_data) < 100 and i % 10 == 0:
#             faces_data.append(resized_img)
            
#         i = i + 1
#         cv2.putText(frame, f"Samples: {len(faces_data)}/100", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2) # Thicker border
        
#     cv2.imshow("Collecting Faces - Press 'q' to quit early", frame)
#     k = cv2.waitKey(1)
    
#     if k == ord('q') or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()


# if len(faces_data) < 100:
#     print(f"Warning: Only {len(faces_data)} samples collected. Attempting to save anyway.")
    


# faces_data = np.asarray(faces_data)

# faces_data = faces_data.reshape(faces_data.shape[0], -1)


# if not os.path.exists('data/names.pkl'):
  
#     names = [name] * faces_data.shape[0] 
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)
#     print(f"Created 'names.pkl' with {len(names)} entries for '{name}'.")
# else:

#     with open('data/names.pkl', 'rb') as f:
#         names = pickle.load(f)
   
#     names = names + [name] * faces_data.shape[0] 
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)
#     print(f"Appended {faces_data.shape[0]} entries for '{name}' to 'names.pkl'. Total names: {len(names)}.")


# if not os.path.exists('data/faces_data.pkl'):
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)
#     print(f"Created 'faces_data.pkl' with {faces_data.shape[0]} samples.")
# else:
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces = pickle.load(f)
#     faces = np.append(faces, faces_data, axis=0)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces, f)
#     print(f"Appended {faces_data.shape[0]} samples to 'faces_data.pkl'. Total faces: {faces.shape[0]}.")

# print("Faces data and names saved successfully!")

import cv2
import pickle
import numpy as np
import os

# Settings
DATA_DIR = 'data'
SAMPLES_PER_PERSON = 100
FACE_SIZE = (50, 50)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load Haarcascade from OpenCV's built-in path
haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(haar_path)

if facedetect.empty():
    print("Error: Could not load Haarcascade.")
    exit()

# Ask for name
name = input("Enter your name: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

# Check if this person already exists
names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

if os.path.exists(names_path):
    with open(names_path, 'rb') as f:
        existing_names = pickle.load(f)
    if name in existing_names:
        print(f"⚠ The name '{name}' already exists in the dataset.")
        choice = input("Do you want to continue adding more samples for this name? (y/n): ").lower()
        if choice != 'y':
            print("Cancelled.")
            exit()
else:
    existing_names = []

# Capture faces
video = cv2.VideoCapture(0)
faces_data = []
i = 0

print(f"Collecting {SAMPLES_PER_PERSON} face samples for '{name}'. Look at the camera.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Cannot access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, FACE_SIZE)

        if len(faces_data) < SAMPLES_PER_PERSON and i % 5 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, f"Samples: {len(faces_data)}/{SAMPLES_PER_PERSON}",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Collecting Faces - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= SAMPLES_PER_PERSON:
        break

video.release()
cv2.destroyAllWindows()

if not faces_data:
    print("No faces collected. Exiting.")
    exit()

# Convert to numpy array
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(faces_data.shape[0], -1)

# Save names
names = existing_names + [name] * faces_data.shape[0]
with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# Save face data
if os.path.exists(faces_path):
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
else:
    faces = faces_data

with open(faces_path, 'wb') as f:
    pickle.dump(faces, f)

print(f"✅ Added {faces_data.shape[0]} samples for '{name}'.")
print(f"📂 Total dataset size: {len(names)} names, {faces.shape[0]} faces.")
