import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import Tuple

# Number of emotion classes and their labels.
NUM_CLASSES = 7
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def load_emotion_pretrained_model() -> nn.Module:
    """
    Loads the pre-trained DeiT model from torch.hub (adapted for emotion recognition).
    This model is assumed to have been fine-tuned on FER-2013.
    """
    print("Loading pre-trained DeiT model from torch.hub (adapted for emotion recognition)...")
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    # Adapt the classification head for FER-2013 (7 classes)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    model.eval()
    print("Model loaded successfully.")
    return model

class EmotionModel:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        # Preprocessing transforms.
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.classes = EMOTION_CLASSES

    def predict(self, image: Image.Image) -> dict:
        x = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        # Convert each value to a Python float.
        return {emotion: float(round(probabilities[idx] * 100, 2)) for idx, emotion in enumerate(self.classes)}


# Global variable to hold the initialized model.
emotion_model = None

def init_inference(device: torch.device):
    """
    Initializes the emotion model. Must be called once at startup.
    """
    global emotion_model
    deit_model = load_emotion_pretrained_model()
    emotion_model = EmotionModel(deit_model, device)
    print("Inference model initialized.")

# Set up Haar Cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_annotate(image: Image.Image) -> Tuple[Image.Image, dict]:
    """
    Accepts a PIL image, detects the largest face, draws a bounding box,
    and returns the annotated image (with only the face highlighted) along
    with the emotion state map.
    """
    if emotion_model is None:
        raise RuntimeError("Model not initialized. Call init_inference() first.")

    # Convert the PIL image to an OpenCV BGR image.
    cv_image = np.array(image.convert("RGB"))
    cv_image = cv_image[:, :, ::-1]  # Convert RGB to BGR
    cv_image = np.ascontiguousarray(cv_image)  # Ensure contiguous memory

    # Convert to grayscale for face detection.
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        x, y, w, h = 10, 10, 50, 50

    # Convert the annotated image back to a PIL image.
    annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # Get the emotion state map.
    emotion_results = emotion_model.predict(image)

    # Do not overlay text on the image (for production use).
    return annotated_image, emotion_results

# if __name__ == "__main__":
#     # For local testing.
#     device = torch.device("xpu")
#     init_inference(device)
#     test_image_path = os.path.join(os.path.dirname(__file__), '..', 'models', '2025_03_01.webp')
#     try:
#         test_image = Image.open(test_image_path).convert("RGB")
#     except Exception as e:
#         print("Error loading test image:", e)
#         sys.exit(1)
#     annotated_image, results = detect_and_annotate(test_image)
#     output_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'annotated_result.webp')
#     annotated_image.save(output_path)
#     print("Emotion detection results:", results)
#     print(f"Annotated image saved to {output_path}")
