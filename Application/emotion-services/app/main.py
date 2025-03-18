import os
import io
import base64
import time
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import inference
import torch
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("xpu")
    start = time.time()
    inference.init_inference(device)
    print(f"All models warmed up in {time.time() - start:.2f} seconds.")
    yield
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Emotion Detection API is running."}

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form("vit_default")
):
    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    print(f"Received image upload request; using model: {model}")
    annotated_image, emotion_results = inference.detect_and_annotate(image, model)
    print(f"Detection completed in {time.time() - start_time:.2f} seconds.")
    
    # Convert annotated image to base64 (WEBP format)
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="WEBP")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print("Sending response to client.")
    return JSONResponse(content={
        "annotated_image": img_base64,
        "emotion_results": emotion_results
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    start_time = time.time()
    model_key = "vit_default"  # or choose dynamically if needed
    while True:
        try:
            # Receive a frame (data URL string)
            data = await websocket.receive_text()
            header, encoded = data.split(',', 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(decoded)).convert("RGB")
            
            # Process the image for emotion detection & annotation.
            annotated_image, emotion_results = inference.detect_and_annotate(image, model_key)
            
            # Re-encode annotated image to WEBP base64 data URL.
            buffered = io.BytesIO()
            annotated_image.save(buffered, format="WEBP")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            annotated_frame = "data:image/webp;base64," + img_base64
            
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            response = {
                "fps": current_fps,
                "frame": annotated_frame,
                "emotion_results": emotion_results
            }
            await websocket.send_text(json.dumps(response))
        except Exception as e:
            print("WebSocket error:", e)
            break


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
