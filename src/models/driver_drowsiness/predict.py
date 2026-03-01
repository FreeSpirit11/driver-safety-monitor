# src/models/driver_drowsiness/predict.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = 'models/mobilenetv2_nthu_finetuned.h5'
TIME_STEPS = 10
IMG_SHAPE = (224, 224, 3)

def load_sequence(sequence_path):
    """
    Load a preprocessed .npy sequence file.
    """
    seq = np.load(sequence_path)
    if seq.shape[0] != TIME_STEPS:
        raise ValueError(f"Expected sequence with {TIME_STEPS} frames, but got {seq.shape[0]}")
    if seq.shape[1:] != IMG_SHAPE:
        raise ValueError(f"Expected frames of shape {IMG_SHAPE}, but got {seq.shape[1:]}")
    # Add batch dimension
    seq = np.expand_dims(seq, axis=0)
    return seq

def predict_sequence(model, sequence):
    """
    Predict drowsiness on a single sequence.
    """
    prediction = model.predict(sequence)
    # Binary classification (0 = alert, 1 = drowsy)
    class_label = int(prediction[0] > 0.5)
    confidence = float(prediction[0])
    return class_label, confidence

def main(sequence_path):
    """
    Main function to load the model, predict, and print the result.
    """
    print("🔍 Loading model...")
    model = load_model(MODEL_PATH)

    print(f"📂 Loading sequence from {sequence_path}")
    sequence = load_sequence(sequence_path)

    print("🚦 Making prediction...")
    label, confidence = predict_sequence(model, sequence)

    label_str = "Drowsy" if label == 1 else "Alert"
    print(f"✅ Prediction: {label_str} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Driver Drowsiness Prediction")
    parser.add_argument("sequence_path", type=str, help="Path to the .npy sequence file")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"🚨 Model file '{MODEL_PATH}' not found. Please train or provide the model first.")
        exit(1)

    if not os.path.isfile(args.sequence_path):
        print(f"🚨 Sequence file '{args.sequence_path}' not found.")
        exit(1)

    main(args.sequence_path)
