# driver-fatigue-monitoring
# 🚗 Driver Fatigue Monitoring System

A deep learning–based system that detects driver drowsiness in real-time using computer vision techniques.  
This project leverages **OpenCV**, **MediaPipe**, and **Machine Learning** to analyze facial landmarks and eye aspect ratio to predict fatigue and alert the driver.

---

##  Overview

Driver fatigue is a leading cause of road accidents. This system monitors the driver's face via a webcam feed and detects signs of fatigue such as:
- Eye closure for a prolonged period
- Yawning
- Head tilt or nodding

Once fatigue is detected, the system triggers an **alert sound** to wake up the driver.

---

##  Tech Stack

- **Python 3.x**
- **OpenCV** – For face and eye detection  
- **MediaPipe** – For facial landmark tracking  
- **NumPy & SciPy** – For mathematical operations  
- **TensorFlow / Keras (optional)** – For model-based fatigue classification  
- **Pygame / playsound** – For alert generation  

---

##  Features

 Real-time face and eye tracking  
 Drowsiness detection using Eye Aspect Ratio (EAR)  
 Audio alert when driver is drowsy  
 Lightweight and fast  
 Works with standard USB webcam  

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/Pranav918-beep1/driver-fatigue-monitoring.git

# Navigate to project folder
cd driver-fatigue-monitoring

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
