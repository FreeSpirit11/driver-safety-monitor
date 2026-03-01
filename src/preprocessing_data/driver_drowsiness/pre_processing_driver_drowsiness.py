import os
import cv2
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

# ========================
# Configuration Constants
# ========================
SEQUENCE_LENGTH_YAWDD = 15
SEQUENCE_LENGTH_NTHU = 10
IMG_SIZE_YAWDD = (64, 64)
IMG_SIZE_NTHU = (224, 224)
BATCH_SIZE = 16
EPOCHS_YAWDD = 22
EPOCHS_NTHU = 12
LEARNING_RATE = 1e-4

# ========================
# Dataset Paths
# ========================
# Root structure
RAW_DATA_ROOT = 'src/data/raw/driver_drowsiness'
PROCESSED_DATA_ROOT = 'src/data/processed/driver_drowsiness'

# YawDD
YAWDD_BASE_PATH = os.path.join(RAW_DATA_ROOT, 'YawDD')
YAWDD_OUTPUT_PATH = os.path.join(PROCESSED_DATA_ROOT, 'YawDD_sequences')
os.makedirs(YAWDD_OUTPUT_PATH, exist_ok=True)

# NTHU
NTHU_OUTPUT_PATH = os.path.join(PROCESSED_DATA_ROOT, 'NTHU_sequences')
DROWSY_DIR = os.path.join(NTHU_OUTPUT_PATH, 'drowsy')
NOTDROWSY_DIR = os.path.join(NTHU_OUTPUT_PATH, 'alert')
os.makedirs(DROWSY_DIR, exist_ok=True)
os.makedirs(NOTDROWSY_DIR, exist_ok=True)

# ========================
# Helper Functions
# ========================
def extract_face_sequence(video_path, label, camera_type, sequence_length, output_path, img_size, face_mesh):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    count = 0
    vid_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                xs = [lm.x for lm in landmarks.landmark]
                ys = [lm.y for lm in landmarks.landmark]

                x_min = max(0, int(min(xs) * w) - 20)
                x_max = min(w, int(max(xs) * w) + 20)
                y_min = max(0, int(min(ys) * h) - 20)
                y_max = min(h, int(max(ys) * h) + 20)

                # Special Dash handling
                if camera_type == 'Dash' and label == -1:
                    mouth_openness = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
                    label = 1 if mouth_openness > 0.1 else 0

                face = frame[y_min:y_max, x_min:x_max]
                if face.size == 0:
                    continue

                face = cv2.resize(face, img_size)
                sequence.append(face)

                if len(sequence) == sequence_length:
                    seq_array = np.array(sequence) / 255.0
                    save_path = os.path.join(output_path, f'{camera_type[:3]}_{label}_{vid_name}_seq{count}.npy')
                    np.save(save_path, seq_array)
                    count += 1
                    sequence = []

    cap.release()

def process_yawdd_dataset(face_mesh):
    for camera_type in ['Mirror', 'Dash']:
        camera_path = os.path.join(YAWDD_BASE_PATH, camera_type)
        if not os.path.exists(camera_path):
            print(f"{camera_path} not found! Skipping...")
            continue

        for gender in ['Male', 'Female']:
            gender_folder = f"{gender}_mirror" if camera_type == 'Mirror' else gender
            gender_path = os.path.join(camera_path, gender_folder)

            if not os.path.exists(gender_path):
                print(f"{gender_path} not found! Skipping...")
                continue

            videos = [f for f in os.listdir(gender_path) if f.endswith('.avi')]
            for video in tqdm(videos, desc=f"Processing {camera_type}/{gender}"):
                video_path = os.path.join(gender_path, video)
                label = 1 if 'yawn' in video.lower() else 0 if camera_type == 'Mirror' else -1
                extract_face_sequence(video_path, label, camera_type, SEQUENCE_LENGTH_YAWDD, YAWDD_OUTPUT_PATH, IMG_SIZE_YAWDD, face_mesh)

# ========================
# NTHU Data Generator
# ========================
class NTHUSequenceGenerator(Sequence):
    def __init__(self, seq_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.seq_paths = seq_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.seq_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.seq_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([np.load(path) for path in batch_paths])
        y = np.array(batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.seq_paths))
            self.seq_paths = [self.seq_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

# ========================
# Dataset Preparation
# ========================
def load_nthu_sequences():
    drowsy_seqs = [os.path.join(DROWSY_DIR, f) for f in os.listdir(DROWSY_DIR) if f.endswith('.npy')]
    alert_seqs = [os.path.join(NOTDROWSY_DIR, f) for f in os.listdir(NOTDROWSY_DIR) if f.endswith('.npy')]
    seq_paths = drowsy_seqs + alert_seqs
    labels = [1] * len(drowsy_seqs) + [0] * len(alert_seqs)
    return train_test_split(seq_paths, labels, test_size=0.2, random_state=42, stratify=labels)

if __name__ == '__main__':
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh

    # Process YawDD
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        print(" Processing YawDD dataset...")
        process_yawdd_dataset(face_mesh)
        print("✅ YawDD sequences saved in:", YAWDD_OUTPUT_PATH)

    # Load NTHU sequences
    print("\n Loading existing NTHU sequences and preparing train/test split...")
    X_train, X_test, y_train, y_test = load_nthu_sequences()
    print(f"✅ Loaded {len(X_train)} training and {len(X_test)} testing NTHU sequences.")
    print("✅ NTHU sequences already exist in:", NTHU_OUTPUT_PATH)

    print("\n Preprocessing complete.")