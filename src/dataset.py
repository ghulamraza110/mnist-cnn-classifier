import numpy as np
import tensorflow as tf
from typing import Tuple


def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and reshapes MNIST data into (N, 28, 28, 1) float32 arrays with one-hot labels."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add single channel dimension
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)

    # One-hot encode labels
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train_one_hot, x_test, y_test_one_hot