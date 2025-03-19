import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import Tuple
from tqdm import tqdm

# Number of emotion classes and their labels.
NUM_CLASSES = 7
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def load_emotion_pretrained_model_local(model_path: str, device: torch.device, optimized: bool = False) -> nn.Module:
    """
    Loads a local ViT-B/16 model for emotion recognition from the given model_path.
    If optimized is True, use a Sequential header with Dropout followed by Linear.
    Otherwise, use a single Linear layer.
    """
    try:
        print(f"Loading local ViT-B/16 model from {model_path} ...")
        model = models.vit_b_16(weights=None)
        if optimized:
            # Use a Sequential header with Dropout (p=0.1) followed by Linear layer.
            model.heads = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(768, NUM_CLASSES)
            )
        else:
            # Use a simple Linear layer.
            model.heads = nn.Linear(768, NUM_CLASSES)
        model = model.to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

class EmotionModel:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
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
        return {emotion: float(round(probabilities[i] * 100, 3)) for i, emotion in enumerate(self.classes)}

# Global dictionary to hold all initialized models.
emotion_models = {}

def init_inference(device: torch.device):
    """
    Initializes and warms up all available emotion models.
    Loads two models: "vit_default" and "vit_optimized".
    """
    global emotion_models
    base_dir = os.path.join(os.path.dirname(__file__), 'models')
    path_default = os.path.join(base_dir, 'vit_b16_fer2013.pth')
    path_optimized = os.path.join(base_dir, 'vit_b16_fer2013_optimized.pth')
    path_resnet50 = os.path.join(base_dir, 'resnet50_fer2013.pth')
    
    try:
        model_default = load_emotion_pretrained_model_local(path_default, device, optimized=False)
        emotion_models["vit_default"] = EmotionModel(model_default, device)
    except Exception as e:
        print(f"Failed to load default model: {e}")

    try:
        model_optimized = load_emotion_pretrained_model_local(path_optimized, device, optimized=True)
        emotion_models["vit_optimized"] = EmotionModel(model_optimized, device)
    except Exception as e:
        print(f"Failed to load optimized model: {e}")
        
    try:
        resnet50_model = load_resnet50_model(path_resnet50, device)
        emotion_models["resnet50"] = ResNetEmotionModel(resnet50_model, device)
    except Exception as e:
        print(f"Failed to load ResNet50 model: {e}")
    
    if not emotion_models:
        raise RuntimeError("No emotion models could be loaded. Check your model files.")
    
    # Warm up each loaded model with a dummy inference using a progress bar.
    dummy = Image.new("RGB", (224, 224), color="white")
    for key in tqdm(emotion_models, desc="Warming up models"):
        try:
            _ = emotion_models[key].predict(dummy)
            print(f"Model '{key}' warmed up.")
        except Exception as e:
            print(f"Error warming up model '{key}': {e}")
    
    print("All available inference models initialized:", list(emotion_models.keys()))

# Set up Haar Cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_annotate(image: Image.Image, model_key: str) -> Tuple[Image.Image, dict]:
    """
    Accepts a PIL image, selects the specified model via model_key,
    detects the largest face, draws a bounding box, and returns the annotated image
    along with the emotion state map.
    """
    if not emotion_models or model_key not in emotion_models:
        raise RuntimeError("Model not initialized or invalid model key. Call init_inference() first.")
    
    selected_model = emotion_models[model_key]
    
    cv_image = np.array(image.convert("RGB"))
    cv_image = cv_image[:, :, ::-1]  # Convert RGB to BGR
    cv_image = np.ascontiguousarray(cv_image)
    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        x, y, w, h = 10, 10, 50, 50

    annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    emotion_results = selected_model.predict(image)
    return annotated_image, emotion_results

def load_resnet50_model(model_path: str, device: torch.device) -> nn.Module:
    print(f"Loading ResNet50 model from {model_path} ...")
    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, NUM_CLASSES)
    resnet50 = resnet50.to(device)
    state_dict = torch.load(model_path, map_location=device)
    resnet50.load_state_dict(state_dict)
    resnet50.eval()
    print("ResNet50 model loaded successfully!")
    return resnet50

class ResNetEmotionModel:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.classes = EMOTION_CLASSES

    def predict(self, image: Image.Image) -> dict:
        x = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        return {emotion: float(round(probabilities[i] * 100, 3)) for i, emotion in enumerate(self.classes)}