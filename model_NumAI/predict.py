"""Load the trained model and predict a handwritten digit."""

# ============================================================
# Imports and model path
# ============================================================
import json
from functools import lru_cache
from pathlib import Path

from model_NumAI.network import forward_pass


MODEL_PATH = Path(__file__).with_name("model.json")


# ============================================================
# Load and use the model
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """Load the layers, weights, and biases from JSON once."""

    # lru_cache(maxsize=1) keeps the result of this argument-free call in memory
    # and avoids rereading the file for every prediction.
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = json.load(f)

    return model["layers"], model["weights"], model["biases"]


def predict(inputs):
    """
    Predict the digit represented by an image of 784 pixels.
    Return the predicted digit, its probability, and the full probability list.
    """

    # The model was trained on 28x28 MNIST images.
    if len(inputs) != 784:
        raise ValueError(
            f"The network expects 784 pixels, got {len(inputs)}"
        )

    layers, weights, biases = load_model()
    probabilities, _ = forward_pass(layers, inputs, weights, biases)

    # Softmax returns a probability distribution; its maximum determines the
    # predicted digit and is used as the confidence score.
    digit = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))

    return digit, float(probabilities[digit]), probabilities