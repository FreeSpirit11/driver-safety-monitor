# predict.py

import cv2
import torch
import numpy as np
from model import FocusLSTM
from feature_extractor import extract_features_from_frame

class DriverAttentivenessPredictor:
    def __init__(self, model_path="models/driver_attentiveness.pth", sequence_length=30):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence = []

        self.model = FocusLSTM().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def update_sequence(self, feature):
        self.sequence.append(feature)
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)

    def predict_focus(self):
        if len(self.sequence) == self.sequence_length:
            input_seq = torch.tensor([self.sequence], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(input_seq)
                pred = torch.argmax(out).item()
                return "Focused" if pred == 1 else "Distracted"
        return None

    def start_video_stream(self):
        cap = cv2.VideoCapture(0)
        print("🔍 Starting Driver Monitoring... Press ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feature = extract_features_from_frame(frame)
            if feature is not None:
                self.update_sequence(feature)
                label = self.predict_focus()

                if label:
                    print(label)
                    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.5, (0, 255, 0), 3)

            cv2.imshow("Driver Monitoring", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Stream ended.")

def main():
    predictor = DriverAttentivenessPredictor()
    predictor.start_video_stream()

if __name__ == "__main__":
    main()
