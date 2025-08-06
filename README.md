<p>
# Face Recognition & Attendance System 🧑‍💻🎯

A Python-based project using OpenCV and KNN to recognize faces in real-time and automatically record attendance into a CSV file. It includes a custom UI layout with a red background area for the webcam display.

---

## 🚀 Features

- 📸 Real-time Face Detection using OpenCV
- 🧠 Face Recognition using K-Nearest Neighbors (KNN)
- 📝 Automatic Attendance Logging into CSV
- 🎨 Custom UI Layout with Red Webcam Background
- 🔊 Voice Feedback using Windows Text-to-Speech (TTS)
- 🖱️ Press `'O'` to take attendance
- 🧪 Press `'Q'` to quit

---

## 🖥️ Requirements

Make sure to install these Python modules:

```bash
pip install opencv-python numpy pywin32
If using KNN or face_recognition:

bash
Copy
Edit
pip install face-recognition scikit-learn
🗂️ Folder Structure
bash
Copy
Edit
face_recognition_project/
├── add_faces.py        
├── train.py              
├── test.py               
├── data/                
├── attendance.csv        
├── trained_model.pkl     
└── README.md             
📷 Screenshot

🛠️ How to Use
Add Faces:

Run add_faces.py

Enter your name

Capture images

Train Model:

Run app.py

Trains and saves the model as faces_data.pkl

Take Attendance:

Run test.py

Press O to take attendance

Press Q to quit

✅ Output
✔️ When you press 'O', your name and timestamp will be saved in attendance.csv.
✔️ The system will also say “Attendance Taken”.

👨‍💻 Developed By
Sujit Malla
Student | Developer | Tech Enthusiast

📬 Contact
For any suggestions or bugs:
📧 Email:sujitmalla000@gmail.com
🔗 GitHub:sujitkumarmalla
💼 LInkedin:http://linkedin.com/in/sujit-kumar-malla-b83248294

⭐ Give a Star
If you like this project, consider giving it a ⭐ on GitHub.
It motivates me to improve it more!



---

## 📤 Steps to Upload on GitHub

1. Create a new repo on GitHub: `face-recognition-attendance`
2. On your local PC:
   ```bash
   git init
   git remote add origin https://github.com/yourusername/face-recognition-attendance.git
   git add .
   git commit -m "Initial commit with full working project"
   git push -u origin main
   </p>