# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np
import cv2
import random

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """Plot training and validation loss (and accuracy if provided)"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if train_accs and val_accs:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, class_names=['Distracted', 'Focused']):
    """Plot confusion matrix and classification report"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def visualize_prediction_frames(model, video_path=None, webcam=True,
                                extractor=None, sequence_length=30,
                                class_names=['Distracted', 'Focused'],
                                n_frames=100):
    """Visualize prediction on live webcam or video frames"""
    cap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0

    while cap.isOpened() and frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        feat = extractor(frame)
        if feat is not None:
            sequence.append(feat)
            if len(sequence) > sequence_length:
                sequence.pop(0)

            if len(sequence) == sequence_length:
                input_tensor = torch.tensor([sequence], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = class_names[pred]
                    color = (0, 255, 0) if pred == 1 else (0, 0, 255)

                    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, color, 3)

        cv2.imshow("Driver Attentiveness", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
