# predict.py

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class FacialExpressionPredictor:
    def __init__(self, model_path="models/best_emotion_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 7)  # 7 facial expression classes assumed
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            return self.class_names[predicted.item()]

    def start_video_stream(self):
        cap = cv2.VideoCapture(0)
        print("📹 Starting Facial Micro-Expression Recognition... Press ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label = self.predict(frame)
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv2.imshow("Facial Expression Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Stream ended.")

def main():
    predictor = FacialExpressionPredictor()
    predictor.start_video_stream()

if __name__ == "__main__":
    main()
