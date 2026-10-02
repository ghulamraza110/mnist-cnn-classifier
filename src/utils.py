import io
import numpy as np
from PIL import Image, ImageOps


def preprocess_image_bytes(image_bytes: bytes, threshold: float = 0.3) -> np.ndarray:
    """
    Converts raw image bytes to a (1, 28, 28, 1) float32 array.
    Automatically handles light-on-dark vs dark-on-light digits.
    """
    # 1. Open image and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 2. Resize to 28x28
    img = img.resize((28, 28))

    img_array = np.array(img, dtype=np.float32) / 255.0

    # If the image has a bright background and dark ink (e.g. pen on paper),
    # invert it so it matches MNIST (bright digit on dark background)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    # 3. Apply thresholding
    binary_array = np.where(img_array > threshold, 1.0, 0.0).astype(np.float32)

    # 4. Reshape to (1, 28, 28, 1)
    processed_tensor = binary_array.reshape(1, 28, 28, 1)
    return processed_tensor