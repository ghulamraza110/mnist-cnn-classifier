import os
from src.dataset import load_and_preprocess_mnist
from src.model import build_mnist_cnn


def train_and_save():
    print("Loading dataset...")
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    print("Building model...")
    model = build_mnist_cnn()
    model.summary()

    print("Training model...")
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Accuracy: {accuracy * 100:.2f}% | Test Loss: {loss:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "mnist_cnn.keras")
    model.save(save_path)
    print(f"Model saved successfully at: {save_path}")


if __name__ == "__main__":
    train_and_save()