"""
Unified Driver Safety & Emotion Analysis Models

- Attentiveness Model: FocusLSTM
- Microexpression Model: MicroexpressionResNet
- TensorFlow Drowsiness Model Loader
- Preprocessing Functions
- Combined Driver State Prediction
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from collections import Counter
from scipy.spatial import distance as dist
import mediapipe as mp
from torchvision import transforms
import tensorflow as tf

# ===============================
# 1️⃣ Models
# ===============================

# 🚗 FocusLSTM (Attentiveness)
class FocusLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2):
        super(FocusLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

class MicroexpressionResNet(nn.Module):
    def __init__(self, num_classes):
        super(MicroexpressionResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
        # ===============================

def load_pytorch_model(model_class, weights_path, device='cpu', **kwargs):
    # Special handling for microexpression model
    if "best_emotion_model.pth" in weights_path:
        num_classes = 7  # or however many you have

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    else:
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    return model

def load_drowsiness_model(model_path):
    """Load drowsiness model from .keras format with fallback"""
    import os
    
    # Try rebuilt model first
    rebuilt_path = model_path.replace(".keras", "_rebuilt.keras")
    if os.path.exists(rebuilt_path):
        try:
            model = tf.keras.models.load_model(rebuilt_path)
            print(f"✅ Loaded rebuilt drowsiness model from {rebuilt_path}")
            return model
        except Exception as e:
            print(f"Warning: Could not load rebuilt model: {str(e)[:100]}")
    
    # Fall back to original path
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found: {model_path}")
        return None
    
    try:
        # Try loading with TensorFlow's load_model
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        error_msg = str(e)
        print(f"Warning: Could not load model from {model_path}")
        print(f"Error details: {error_msg[:200]}")
        
        # Return None and let predict function handle it
        return None

# ===============================
# 3️⃣ Preprocessing
# ===============================

# 3a. Feature Extraction (Attentiveness)
mp_face_mesh = mp.solutions.face_mesh

# Initialize FaceMesh once globally to avoid threading issues
try:
    face_mesh_instance = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception as e:
    print(f"Warning: FaceMesh initialization failed: {e}")
    face_mesh_instance = None

def extract_features_from_frame(frame):
    if face_mesh_instance is None:
        return np.zeros(4)
    
    try:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_instance.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            def to_px(lm): return np.array([lm.x * w, lm.y * h])
            nose = to_px(landmarks[1])
            left_eye = to_px(landmarks[33])
            right_eye = to_px(landmarks[263])
            left_iris = to_px(landmarks[468])
            right_iris = to_px(landmarks[473])
            eye_center = (left_eye + right_eye) / 2
            iris_center = (left_iris + right_iris) / 2
            gaze = iris_center - eye_center
            head = nose - eye_center
            gaze /= np.linalg.norm(gaze) + 1e-6
            head /= np.linalg.norm(head) + 1e-6
            return np.concatenate([gaze, head])
        else:
            return np.zeros(4)
    except Exception as e:
        print(f"Warning: Feature extraction failed: {e}")
        return np.zeros(4)

# 3b. Microexpression Preprocessing
transform_microexpression = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def preprocess_frame_for_microexpression(frame):
    return transform_microexpression(frame).unsqueeze(0)

# 3c. Drowsiness Preprocessing
def preprocess_frame_for_drowsiness(frame, img_size=(224, 224)):
    img = cv2.resize(frame, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

# ===============================
# 4️⃣ EAR for Drowsiness
# ===============================

EAR_THRESHOLD = 0.2
def get_ear(landmarks, frame_w, frame_h):
    left_indices = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]
    def eye_aspect_ratio(eye_pts):
        A = dist.euclidean(eye_pts[1], eye_pts[5])
        B = dist.euclidean(eye_pts[2], eye_pts[4])
        C = dist.euclidean(eye_pts[0], eye_pts[3])
        return (A + B) / (2.0 * C)
    left_eye = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in left_indices]
    right_eye = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in right_indices]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0

# ===============================
# 5️⃣ Model Predictors
# ===============================

def predict_driver_attentiveness(model, frames, device='cpu'):
    features = [extract_features_from_frame(frame) for frame in frames]
    input_seq = torch.tensor([features], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_seq)
    pred = torch.argmax(output).item()
    return "Focused" if pred == 1 else "Distracted"

def predict_microexpression(model, frames, device='cpu'):
    emotion_names = {
        0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprised",
        4: "Angry", 5: "Disgust", 6: "Fear"
    }
    predictions = []
    for frame in frames:
        input_img = preprocess_frame_for_microexpression(frame).to(device)
        with torch.no_grad():
            output = model(input_img)
        pred_class = torch.argmax(output).item()
        predictions.append(pred_class)
    final_pred = Counter(predictions).most_common(1)[0][0]
    emotion_name = emotion_names.get(final_pred, f"Unknown ({final_pred})")
    return emotion_name

def predict_driver_drowsiness(model, frames, fps=30, sequence_length=15):
    # Handle case where model failed to load
    if model is None:
        # Use eye aspect ratio as fallback
        print("Warning: Drowsiness model not available, using EAR-only detection")
        closed_eye_frame_count = 0
        frame_w, frame_h = frames[0].shape[1], frames[0].shape[0]
        
        if face_mesh_instance is None:
            return f"Driver is ALERT (No face detection available)"
        
        try:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh_instance.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    ear = get_ear(landmarks, frame_w, frame_h)
                    if ear < EAR_THRESHOLD:
                        closed_eye_frame_count += 1
            eyes_closed_seconds = closed_eye_frame_count / fps
            if eyes_closed_seconds >= 1.0:
                return f"Driver is DROWSY (EAR-based, {eyes_closed_seconds:.1f}s closed)"
            else:
                return f"Driver is ALERT (EAR-based, {eyes_closed_seconds:.1f}s closed)"
        except Exception as e:
            print(f"Warning: EAR detection failed: {e}")
            return f"Driver is ALERT (Detection unavailable)"
    
    closed_eye_frame_count = 0
    frame_w, frame_h = frames[0].shape[1], frames[0].shape[0]
    
    if face_mesh_instance is not None:
        try:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh_instance.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    ear = get_ear(landmarks, frame_w, frame_h)
                    if ear < EAR_THRESHOLD:
                        closed_eye_frame_count += 1
        except Exception as e:
            print(f"Warning: EAR calculation failed: {e}")
    
    eyes_closed_seconds = closed_eye_frame_count / fps
    sequences = []
    for i in range(0, len(frames) - sequence_length + 1, sequence_length):
        chunk = frames[i:i + sequence_length]
        processed = [preprocess_frame_for_drowsiness(f) for f in chunk]
        sequences.append(processed)
    class_votes = {"DROWSY": 0, "ALERT": 0}
    confidences = []
    for seq in sequences:
        input_batch = np.expand_dims(seq, axis=0)
        preds = model.predict(input_batch)[0]
        if len(preds) == 2:
            class_idx = np.argmax(preds)
            conf = float(preds[class_idx])
            model_class = "DROWSY" if class_idx == 1 else "ALERT"
        else:
            model_class = "DROWSY" if preds[0] > 0.5 else "ALERT"
            conf = float(preds[0]) if preds[0] > 0.5 else 1 - preds[0]
        class_votes[model_class] += 1
        confidences.append(conf)
    voted_class = max(class_votes, key=class_votes.get)
    avg_conf = np.mean(confidences) if confidences else 0.5
    if eyes_closed_seconds >= 1.0 or voted_class == "DROWSY":
        final_class = "DROWSY"
    else:
        final_class = "ALERT"
    return f"Driver is {final_class} ({avg_conf*100:.2f}%)"

# ===============================
# 6️⃣ Combined Prediction
# ===============================

def combined_driver_state(att_model, microexp_model, drowsy_model, frames, device='cpu'):
    attentiveness = predict_driver_attentiveness(att_model, frames, device)
    microexp = predict_microexpression(microexp_model, frames, device)
    drowsiness = predict_driver_drowsiness(drowsy_model, frames)
    return {
        "Driver Attentiveness": attentiveness,
        "Driver Microexpression": microexp,
        "Driver Drowsiness": drowsiness
    }

# ===============================
# 7️⃣ Main Execution
# ===============================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    att_model = load_pytorch_model(FocusLSTM, "../notebooks/driver_attentiveness/model.pth", device=device, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2)
    microexp_model = load_pytorch_model(MicroexpressionResNet, "../notebooks/facial_micro_expression/best_emotion_model.pth", device=device, num_classes=7)
    drowsy_model = load_drowsiness_model("../notebooks/driver_drowsiness/mobilenet_model.keras")
    cap = cv2.VideoCapture(0)
    frames = []
    while len(frames) < 30:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    result = combined_driver_state(att_model, microexp_model, drowsy_model, frames, device)
    print(result)
