# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import numpy as np
import cv2


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """Plot training and validation loss (and accuracy if provided)."""
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy
    if train_accs and val_accs:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix and print classification report."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


def visualize_predictions(model, transform, class_names, video_path=None, webcam=True, n_frames=100):
    """
    Live or video prediction visualizer for Facial Micro-Expression.
    Args:
        model: Trained PyTorch model.
        transform: Preprocessing transform to apply to frames.
        class_names: List of class labels.
        video_path: Path to video file. If None, webcam is used.
        webcam: Bool, whether to use webcam or not.
        n_frames: Number of frames to process before stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture(video_path)
    frame_count = 0

    print("🎥 Starting video prediction... Press ESC to quit.")

    while cap.isOpened() and frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        img = cv2.resize(frame, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack([img] * 3, axis=-1)  # Convert to 3 channels
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()
            label = class_names[pred]

        # Display
        cv2.putText(orig, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Facial Micro-Expression Prediction", orig)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended.")
