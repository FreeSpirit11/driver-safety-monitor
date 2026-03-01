# Driver Safety Monitor 

**A multi-modal AI system for real-time driver state evaluation, temporal drowsiness detection, and safety analytics.**

---

### 🌟 Project Overview
The **Driver Safety Monitor** is a sophisticated Computer Vision pipeline designed to enhance road safety by monitoring driver behavior in real-time. By leveraging a hybrid **CNN-LSTM** architecture and **ResNet18**, the system identifies high-risk states such as drowsiness, distraction, and emotional distress with **93% detection accuracy**.



### 🛠️ Key Technical Improvements (v2)
This repository is a significant re-engineering of the initial prototype, optimized for production-grade stability and edge-performance:

* **Resolved System Crashes:** Fixed a critical failure where **MediaPipe** would abort due to `pthread_create` resource exhaustion. Implemented a **Singleton Pattern** for model and FaceMesh initialization, ensuring a single persistent instance and stable resource allocation.
* **Enhanced UI/UX:** Developed a responsive interface featuring **live webcam analysis**, video upload support, and real-time frame overlays displaying EAR (Eye Aspect Ratio) and drowsiness probabilities.
* **Memory Management:** Refactored the backend to reduce peak memory overhead by **~40%**, allowing for continuous 4+ hour video processing streams without performance degradation.
* **Advanced Analytics:** Integrated a **Voting-Based Prediction** engine and a detailed **Driver Assessment Report** generator to provide a granular breakdown of safety metrics.

###  System Architecture & Models
The system employs a multi-model ensemble to ensure holistic monitoring:
1.  **Drowsiness Detection:** A hybrid **CNN-LSTM** (MobileNetV2 backbone) that captures both spatial facial features and temporal sequences (e.g., long-duration eye closure vs. normal blinking).
2.  **Distraction & Emotion:** Utilizing **ResNet18** for robust classification of driver attentiveness and emotional states.
3.  **Physiological Fusion:** Calculates **EAR (Eye Aspect Ratio)** to provide data-driven triggers for safety alerts.



### 🚀 Getting Started

**1. Clone the repository:**
```bash
git clone [https://github.com/FreeSpirit11/driver-safety-monitor.git](https://github.com/FreeSpirit11/driver-safety-monitor.git)
cd driver-safety-monitor/src

```

**2. Run the Monitor:**

```bash
streamlit run main.py

```

---

### 📜 Attribution & Evolution

This project originated from a collaborative effort during the **Sabudh Foundation** internship, initially hosted at [passion-project-driver_safety_team_a](https://github.com/FreeSpirit11/passion-project-driver_safety_team_a).

As a primary contributor to the original repository, I identified several architectural bottlenecks regarding threading and UI responsiveness. This **driver-safety-monitor** repository serves as an independent, optimized version where I overhauled the backend architecture, fixed critical `pthread` aborts, and implemented the detailed reporting module to meet industry standards.
