# Driver Safety Monitoring Project

This document provides a comprehensive description of the driver safety system as it currently stands. 

---

## 📁 Repository Structure

```
DockerFile
FINAL_STATUS.md
LICENSE
PRODUCTION_READY.md
README.md
requirements.txt
notebooks/
  driver_attentiveness/
  driver_drowsiness/
  facial_micro_expression/
reports/
src/
venv/
```

- `DockerFile` - (unused) placeholder for containerization.
- `PRODUCTION_READY.md` - checklist and environment notes for production deployment.
- `README.md` - quickstart instructions and overview.
- `requirements.txt` - Python dependencies required to run the system.

### `notebooks/`
Contains the exploratory and training notebooks used to build each model:

- `driver_attentiveness/` - utilities for preparing data and training the FocusLSTM model. Key scripts: `create_dataset.py`, `feature_extractor.py`, `train.py`.
- `driver_drowsiness/` - primary notebook `driver_drowsiness_detection.ipynb` where the MobileNetV2+LSTM model is defined, trained (YawDD and NTHU datasets), and real-time tested. The fine‑tuned model file (`fine_tuned_model.keras`) is saved here along with the original packaged `mobilenet_model.keras`.
- `facial_micro_expression/` - contains code for training and running a ResNet18 emotion classifier. The best model `best_emotion_model.pth` is stored here.

### `reports/`
Quarto project with the final report, figures, and presentations. Not needed for runtime but useful for documentation.

### `src/`
Primary source code for the application.

```
src/
  backend.py
  main.py
  data/
  feature_engineering/
  models/
  preprocessing_data/
  visualization/
```

#### `backend.py`
- Implements model classes (`FocusLSTM`, `MicroexpressionResNet`) and helper functions.
- Contains image preprocessing, feature extraction (MediaPipe FaceMesh), EAR calculation.
- Defines prediction wrappers for each model and a `combined_driver_state` helper.
- Manages global MediaPipe instance for thread safety.

#### `main.py`
- Streamlit application orchestrating UI, video processing, and inference.
- Defines CSS theme, header, sidebar behavior (now minimal), and layout.
- Loads models via `load_models()` and captures webcam/video frames.
- Implements chunking mechanism (30-frame chunks for attention; internal 10-frame segments for drowsiness voting).
- Renders results, metrics, plots, and handles CSV export.

#### `data/`
- Contains directories for raw, interim, processed datasets for each task. Used during preprocessing but not required for inference.

#### `feature_engineering/` and `preprocessing_data/`
- Scripts to transform raw video frames into model-ready sequences (e.g., computing yawning/mouth features for drowsiness or gaze/head features for attentiveness).
- Each subfolder contains task-specific code.

#### `models/`
- Contains training and prediction utilities for each subproblem; not used by the Streamlit app but useful for offline experimentation.

#### `visualization/`
- Plotting functions for exploratory analysis (again, mostly used during development).

### Other Files

- `venv/` - Python virtual environment (not version-controlled).

---

## 🛠️ Workflow Overview

1. **Data Preparation**
   - Raw videos placed in `src/data/raw/…`.
   - Preprocessing scripts read raw files, extract frames, compute landmarks, and save processed `.npy` sequences.
   - Feature engineering transforms these sequences into training datasets for each model.

2. **Model Training**
   - Each model has its own notebook where training hyperparameters are configured and figures are generated.
   - After training, models are saved back into `notebooks/…` and can be copied into production if needed.

3. **Real-Time Inference**
   - The Streamlit app (`src/main.py`) loads three models at startup and configures the inference environment.
   - Video input (webcam or file) is captured at 30 FPS.
   - Frames are buffered into a sliding window; chunks of 30 frames are processed.
   - Predictions:
     - **Attentiveness**: Pass entire 30-frame chunk to FocusLSTM.
     - **Microexpression**: Predict per-frame with ResNet, vote mode.
     - **Drowsiness**: Divide chunk into three 10-frame segments, run fine‑tuned model on each, vote results.
   - Results are overlaid on the video and displayed in the dashboard with metrics and plots.

4. **Deployment**
   - Launch the app via `streamlit run src/main.py`.
   - For production, ensure environment matches `requirements.txt` and GPU capability (optional) is set up.

---

## 📦 Dependencies
Install via:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key libraries: Streamlit, OpenCV, PyTorch, TensorFlow/Keras, MediaPipe, NumPy, Plotly.

---

## 🔍 Current Project State

- UI: grey theme, fixed controls, overlay mandatory, start/stop buttons centered.
- Models: Attentiveness LSTM, Drowsiness CNN‑LSTM (10-frame, user‑trained), Emotion ResNet18.
- Inference: chunk‑based, voting for drowsiness, risk score calculation present but can be disabled.
- Documentation: comprehensive handover information in `FINAL_STATUS.md` and this file.

All unnecessary experimental artifacts have been removed. The repo is now clean and ready for handoff.

---

## 📌 Handover Notes

- The drowsiness model is essential and must remain at 10 frames; changing sequence lengths requires retraining.
- If you wish to retrain or replace models, use the notebooks under `notebooks/…` as starting points.
- `src/backend.py` contains all preprocessing logic; adjust EAR threshold or feature extractors here if needed.
- To modify the UI or add features, edit `src/main.py` and ensure new CSS stays within grey palette guidelines.

Feel free to reach out for clarifications—this document should allow a newcomer to quickly understand and extend the system.