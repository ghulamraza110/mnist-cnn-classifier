import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

from src.utils import preprocess_image_bytes

MODEL_PATH = os.path.join("models", "mnist_cnn.keras")
loaded_model: tf.keras.Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. Please run `python train.py` first."
        )
    print(f"Loading model from {MODEL_PATH}...")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    yield
    print("Shutting down service...")


app = FastAPI(
    title="MNIST Digit Classifier API",
    description="Upload a handwritten digit image to get prediction from CNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "MNIST Digit Classifier API is running. Go to /docs to test endpoints."}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {list(allowed_extensions)}",
        )

    try:
        image_bytes = await file.read()
        tensor_input = preprocess_image_bytes(image_bytes)

        # Run inference
        predictions = loaded_model.predict(tensor_input, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100.0

        all_probabilities = {str(digit): round(float(prob), 4) for digit, prob in enumerate(predictions)}

        return {
            "filename": file.filename,
            "predicted_digit": predicted_digit,
            "confidence_percentage": round(confidence, 2),
            "probabilities": all_probabilities,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")