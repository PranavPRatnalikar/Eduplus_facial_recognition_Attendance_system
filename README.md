# Eduplus Facial Recognition Attendance System
##  📥 Download Pre-Trained Model

[Drive Link](https://drive.google.com/file/d/1dxhQmfZ3n83crHHxlhd2hiJl3y6XQB8t/view?usp=sharing)


## Firebase Setup
🔥 Setting Up Firebase Database

Follow these steps to create and configure your Firebase Realtime Database:

1️⃣ Create a Firebase Project

1. Go to Firebase Console

2. Click "Add Project", give it a name, and continue with default settings.

3. Enable Firebase Realtime Database:

 - In the Firebase dashboard, go to Build → Realtime Database

 - Click "Create Database"

 - Select a region and choose "Start in Test Mode" (for development purposes)

2️⃣ Get Firebase Admin SDK JSON File

1. In Firebase Console, navigate to Project Settings → Service Accounts.

2. Click Generate new private key.

3. Download the .json file and place it inside your project directory.

4. Add the file path inside your code wherever Firebase authentication is required.

3️⃣ Get Firebase Database URL

1. In Firebase Console, go to Realtime Database.

2. Copy the database URL (it looks like https://your-project-id.firebaseio.com/)

3. Add this URL in app.py .


## Installation

1️⃣ Clone the Repository

```bash
  git clone https://github.com/PranavPRatnalikar/Eduplus_facial_recognition_Attendance_system.git
cd Eduplus_facial_recognition_Attendance_system
```

2️⃣ Install Dependencies

Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

3️⃣ Run the Application
```bash
streamlit run app.py
```


    

## 🛠️ Features

Face recognition-based attendance marking.

Successfully detecting and recognizing 15-20 individuals in a single input image

Firebase integration for data storage.

The system efficiently extracts facial features using face_recognition and compares them against a pre-stored database of known individuals.

Recognized individuals are automatically labeled with their names.

For those who are detected but not recognized, the system provides an option for manual verification.

