# 📱 Attendance Management System Using Face Recognition

An AI-powered attendance solution that automates student attendance using facial recognition — eliminating manual roll calls with real-time face detection on Android.

---

## 📌 Project Overview

This system combines an **Android mobile app** with a **Flask backend server** and a deep learning face recognition pipeline to identify students and mark attendance automatically — from both live camera feeds and group classroom photos.

### System Components
- Android Application (Frontend)
- Flask Backend Server (API)
- FaceNet Face Recognition Model
- MTCNN Face Detection
- Student Database Management
- Attendance Report Generation

---

## ✨ Features

### 👤 Student Management
- Add, edit, and delete student records
- View all registered students

### 🧠 Face Enrollment
- Capture multiple face samples per student
- Automatic face embedding generation using FaceNet
- Secure embedding storage

### 📷 Live Attendance
- Real-time face detection via device camera
- Real-time face recognition
- Automatic attendance marking on recognition

### 👥 Group Attendance
- Capture classroom group photos
- Detect and recognize multiple faces simultaneously
- Mark attendance for all identified students at once

### 📊 Attendance Management
- Daily attendance records with date and time
- Attendance history tracking
- CSV export support

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Android Frontend | Android Studio, Java, CameraX, XML |
| Backend Server | Python, Flask |
| Face Detection | MTCNN |
| Face Recognition | FaceNet, TensorFlow |
| Image Processing | OpenCV |
| Database | Pickle, CSV Files |

---

## 🔄 System Workflow

```
1. Admin adds student details
        ↓
2. Student face samples are captured
        ↓
3. FaceNet generates face embeddings
        ↓
4. Embeddings stored in database
        ↓
5. Live camera / group photo captured
        ↓
6. MTCNN detects faces → FaceNet recognizes them
        ↓
7. Attendance marked automatically
        ↓
8. Records saved as CSV
```

---

## 📦 Modules

### Student Management Module
Maintains student records including Name, Enrollment Number, Branch, Semester, and Section.

### Face Enrollment Module
Captures multiple face samples per student for improved recognition accuracy and generates FaceNet embeddings.

### Face Recognition Module
Uses FaceNet embeddings and cosine similarity to identify students against the stored database.

### Attendance Module
Automatically records attendance with date and time stamps on successful recognition.

### Report Module
Provides session attendance records and downloadable CSV reports for review and analysis.

---

## ✅ Advantages

- Contactless attendance marking
- Time-efficient compared to manual roll calls
- High recognition accuracy with multi-sample enrollment
- Easy student management via Android interface
- Reduced human errors
- Group attendance support for large classrooms
- Scalable architecture (Android + Flask backend)

---

## 🔮 Future Enhancements

- [ ] Cloud database integration
- [ ] Teacher login system
- [ ] Attendance analytics dashboard
- [ ] SMS / Email notifications for low attendance
- [ ] Multi-class and multi-subject support
- [ ] Anti-spoofing detection

---

## 📁 Project Structure

```
face-attendance-application/
│
├── android/                  # Android Studio project
│   ├── app/src/main/
│   │   ├── java/             # Java source files
│   │   └── res/xml/          # UI layouts
│
├── backend/                  # Flask server
│   ├── app.py                # Main Flask application
│   ├── requirements.txt      # Python dependencies
│   └── student_embeddings.pkl
│
├── model/
│   └── facenet.tflite        # TFLite model for Android
│
├── attendance/               # Generated CSV reports
│   ├── master_attendance.csv
│   └── attendance_report.csv
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Backend (Flask Server)

```bash
# Clone the repository
git clone https://github.com/Mayankpawar28/face-attendance-application.git
cd face-attendance-application/backend

# Install dependencies
pip install flask keras-facenet mtcnn opencv-python pandas scikit-learn tensorflow

# Run the server
python app.py
```

### Android App
1. Open the `android/` folder in Android Studio
2. Update the Flask server IP in the network config
3. Build and run on your Android device

---

## 👤 Developed By

**Himesh Rinchhodiya**  
Computer Science and Business Systems (CSBS)  
Medicaps University

**Mayank Pawar**  
Computer Science and Engineering (CSE)  
Medicaps University  
GitHub: [github.com/Mayankpawar28](https://github.com/Mayankpawar28)

---

## 📄 License

This project is developed for academic and educational purposes.
