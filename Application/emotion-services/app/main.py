import os
import io
import base64
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import inference  # our inference module with init_inference() and detect_and_annotate()
import torch
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("xpu")
    start = time.time()
    inference.init_inference(device)
    # Warm up the model with a dummy image (e.g., a blank image)
    from PIL import Image
    dummy = Image.new("RGB", (224, 224), color="white")
    _ = inference.detect_and_annotate(dummy)
    print(f"Model initialized and warmed up in {time.time() - start:.2f} seconds.")
    yield
    print("Shutting down.")


app = FastAPI(lifespan=lifespan)

# Configure CORS: allow requests from the frontend (e.g., http://localhost:3000)
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
async def upload_image(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    print("Received image upload request.")
    annotated_image, emotion_results = inference.detect_and_annotate(image)
    print(f"Detection and annotation completed in {time.time() - start_time:.2f} seconds.")
    
    # Convert the annotated image to a base64-encoded string (WEBP format).
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="WEBP")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    print("Sending response to client.")
    return JSONResponse(content={
        "annotated_image": img_base64,
        "emotion_results": emotion_results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
