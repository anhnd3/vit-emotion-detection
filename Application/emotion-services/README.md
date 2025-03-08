# 📡 Emotion Detection Backend Services

Backend service for the emotion recognition system, built using FastAPI and Vision Transformer (ViT) model.

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **FastAPI** (high-performance API framework)
- **PyTorch** (deep learning framework optimized with Intel XPU support)
- **Vision Transformer (ViT)** pre-trained model (via HuggingFace Transformers)

---

## 🚀 Getting Started

### ✅ Step 1: Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

## 📦 Step 2: Install Dependencies

Install required Python packages from the provided requirements.txt:

```bash
pip install -r requirements.txt
```

### 📦 Key Dependencies

- **FastAPI:** `fastapi==0.115.11`
- **Uvicorn:** `uvicorn==0.34.0`
- **PyTorch (Intel XPU build):** `torch==2.7.0.dev20250228+xpu`
- **TorchVision:** `torchvision==0.22.0.dev20250228+xpu`
- **Transformers (HuggingFace):** `transformers==4.48.3`
- **OpenVINO Toolkit:** `openvino==2025.0.0`
- **NumPy:** `numpy==2.1.3`
- **Pandas:** `pandas==2.2.3`
- **OpenCV:** `opencv-python==4.11.0.86`
- **Matplotlib:** `matplotlib==3.10.0`

*(Refer to `requirements.txt` for the complete list of dependencies.)*

## ▶️ Step 3: Run the Application

Install required Python packages from the provided requirements.txt:

```bash
cd emotion-services
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📃 API Endpoint

### 🔹 Upload and Predict Emotion

- **URL:** `/upload`
- **Method:** `POST`
- **Description:** Uploads an image file and returns the predicted emotion and confidence score.

**Request Example (using curl):**

```bash
curl -X POST "http://localhost:8000/upload" \
-H "Content-Type: multipart/form-data" \
-F "image=@path_to_your_image.jpg"
```

### 🔹 Response Example (JSON)

```json
{
    "annotated_image": "{base64_image_data}",
    "emotion_results": {
        "angry": 12.109999656677246,
        "disgust": 20.690000534057617,
        "fear": 9.0,
        "happy": 7.289999961853027,
        "neutral": 15.529999732971191,
        "sad": 17.110000610351562,
        "surprised": 18.270000457763672
    }
}
```

> Note: The annotated_image field returns a base64-encoded string of the annotated image.
