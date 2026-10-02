# MNIST Handwritten Digit Classifier (CNN & FastAPI)

A modular, production-ready machine learning project implementing a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify handwritten digits (0–9) using the MNIST dataset, served via a FastAPI inference backend.

---

## Project Structure

```text
mnist-cnn-classifier /
│
├── models/
│   └── mnist_cnn.keras         # Saved model artifact (generated after training)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Data loader, normalization, and shape handling
│   ├── model.py                # CNN model architecture definition
│   └── utils.py                # Image conversion, thresholding, and tensor formatting
│
├── train.py                    # Script to train, evaluate, and save the model
├── app.py                      # FastAPI application for inference
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```

---

## Features

* **End-to-End CNN Pipeline**: Multi-layer Convolutional blocks (`Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`) designed specifically for spatial feature extraction.
* **Robust Preprocessing**: Automatically handles image inversion (light ink on dark paper vs. dark ink on light paper), binary thresholding, and dimension alignment to `(1, 28, 28, 1)`.
* **FastAPI Web Service**: High-performance asynchronous endpoint with Swagger UI documentation for live image uploads.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd mnist-cnn-classifier

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
python -m pip install tensorflow fastapi uvicorn pillow numpy python-multipart

```

---

## Usage

### 1. Train the CNN Model
`Note:` the file is already included in the repository, but you can retrain the model if needed.

Run `train.py` to download the MNIST dataset, train the network for 10 epochs, evaluate test accuracy, and export the model to `models/mnist_cnn.keras`:

```bash
python train.py

```

### 2. Start the FastAPI Server

Start the Uvicorn server:

```bash
python -m uvicorn app:app --reload  

```

The application will be accessible at:

* **Base URL**: `http://localhost:8000`
* **Interactive Swagger Docs**: `http://localhost:8000/docs`
* **ReDoc**: `http://localhost:8000/redoc`

---

## API Reference

### Health Check

* **Endpoint**: `GET /`
* **Response**:

```json
{
  "message": "MNIST Digit Classifier API is running. Go to /docs to test endpoints."
}

```

### Predict Digit

* **Endpoint**: `POST /predict`
* **Content-Type**: `multipart/form-data`
* **Payload**: `file` (Image formats supported: `.jpg`, `.jpeg`, `.png`, `.bmp`)



---

## Testing via Browser (Swagger UI) 

1. Navigate to `http://localhost:8000/docs`.
2. Expand the `POST /predict` endpoint.
3. Click **Try it out**.
4. Choose an image file of a digit from your system.
5. Click **Execute** to view the response payload and prediction confidence.

#### Example Response

```json
{
  "filename": "digit_image.png",
  "predicted_digit": 5,
  "confidence_percentage": 99.87,
  "probabilities": {
    "0": 0.0,
    "1": 0.0,
    "2": 0.0,
    "3": 0.0001,
    "4": 0.0,
    "5": 0.9987,
    "6": 0.0005,
    "7": 0.0,
    "8": 0.0006,
    "9": 0.0001
  }
}

```
