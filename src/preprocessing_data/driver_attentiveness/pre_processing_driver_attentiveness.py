import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from feature_extractor import extract_features_from_frame

warnings.filterwarnings('ignore')

# ========================
# Configuration Constants
# ========================
SEQUENCE_LENGTH = 30
RAW_DATA_PATH = "src/data/raw/driver_attentiveness"
PROCESSED_DATA_PATH = "src/data/processed/driver_attentiveness"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# ========================
# Preprocessor Class
# ========================
class DriverAttentivenessPreprocessor:
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length

    def extract_sequences(self, video_path, label):
        cap = cv2.VideoCapture(video_path)
        features = []
        sequences = []
        labels = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feat = extract_features_from_frame(frame)
            if feat is not None:
                features.append(feat)
                if len(features) >= self.sequence_length:
                    sequence = np.array(features[-self.sequence_length:])
                    sequences.append(sequence)
                    labels.append(label)

        cap.release()
        return sequences, labels

    def process_videos_in_directory(self):
        all_sequences = []
        all_labels = []

        for class_dir in os.listdir(RAW_DATA_PATH):
            class_path = os.path.join(RAW_DATA_PATH, class_dir)
            if not os.path.isdir(class_path):
                continue

            label = 1 if class_dir.lower() == "focused" else 0  # Adjust this logic to your class folders
            videos = [f for f in os.listdir(class_path) if f.endswith(".mp4") or f.endswith(".avi")]

            for video in tqdm(videos, desc=f"Processing {class_dir} videos"):
                video_path = os.path.join(class_path, video)
                sequences, labels = self.extract_sequences(video_path, label)
                all_sequences.extend(sequences)
                all_labels.extend(labels)

        return np.array(all_sequences), np.array(all_labels)

    def save_sequences(self, X, y):
        np.save(os.path.join(PROCESSED_DATA_PATH, "X_sequences.npy"), X)
        np.save(os.path.join(PROCESSED_DATA_PATH, "y_labels.npy"), y)
        print(f"✅ Saved {len(X)} sequences to {PROCESSED_DATA_PATH}")


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    preprocessor = DriverAttentivenessPreprocessor()
    X, y = preprocessor.process_videos_in_directory()
    preprocessor.save_sequences(X, y)
