"""Train the network on MNIST and save the resulting model."""

# ============================================================
# Imports and configuration
# ============================================================
import json
from pathlib import Path

from model_NumAI.network import train
from model_NumAI.mnist import load_mnist, load_mnist_test


# Architecture: 784 input pixels, two hidden layers of 64 neurons,
# and one output for each possible digit (0 to 9).
LAYERS = [64, 64, 10]

MODEL_PATH = Path(__file__).with_name("model.json")


# ============================================================
# Save the model
# ============================================================

def save_model(weights, biases):
    """Save the architecture, weights, and biases to a JSON file."""

    model = {
        "layers": LAYERS,
        "weights": weights,
        "biases": biases
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f)


# ============================================================
# Training
# ============================================================

def main():
    """Load the data, train the network, and save its parameters."""

    # Load the training and test sets.
    X, labels = load_mnist()
    X_test, labels_test = load_mnist_test()

    # One-hot encoding represents the expected class as a vector:
    # the correct digit is 1 and all other entries are 0.
    Y = []

    for label in labels:
        target = [0.0] * 10
        target[label] = 1.0
        Y.append(target)

    # An epoch is one complete pass over the training data.
    # The callback saves the parameters at the end of each pass.
    train(
        X,
        Y,
        LAYERS,
        epochs=10,
        lr=0.01,
        test_data=(X_test, labels_test),
        on_epoch=save_model
    )

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()