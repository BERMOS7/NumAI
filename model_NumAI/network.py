"""
Fully connected neural network implemented from scratch.
Softmax output and cross-entropy loss for classification.
Vectorized with NumPy for faster computation.
"""

# ============================================================
# Imports
# ============================================================
import math
import random
import time

import numpy as np

# ============================================================
# Hyperparameters
# ============================================================

LR = 0.01       # Learning rate
EPOCHS = 10     # Number of complete passes over the dataset

# ============================================================
# Initialization
# ============================================================

def generate_parameters(layer_neuron, inputs):
    """
    Create initial weights and biases for all layers.
    - Weights are sampled from a normal distribution centered at 0, with a
      standard deviation based on the layer size (see below).
    - Biases are initialized to 0.
    """

    weights = []  # weights[layer]: matrix (neurons x connections)
    biases = []   # biases[layer]: vector (neurons)

    for layer_i in range(len(layer_neuron)):

        # Input size for the first layer; otherwise, the previous layer's size
        if layer_i == 0:
            n_inputs = len(inputs)
        else:
            n_inputs = layer_neuron[layer_i - 1]

        # Choose a weight standard deviation that keeps signal magnitudes stable
        if layer_i == len(layer_neuron) - 1:
            std = math.sqrt(2 / (n_inputs + layer_neuron[layer_i]))
        else:
            std = math.sqrt(2 / n_inputs)

        # One random weight for each incoming connection to each neuron
        weights.append(np.random.normal(0, std, (layer_neuron[layer_i], n_inputs)))

        # Zero biases help avoid inactive ReLU neurons at initialization
        biases.append(np.zeros(layer_neuron[layer_i]))

    return weights, biases


# ============================================================
# Forward propagation
# ============================================================

def neuron(inputs, weights, bias):
    """Compute a neuron's output: weighted sum of inputs plus bias."""

    return np.dot(inputs, weights) + bias


def softmax(z):
    """Convert raw scores into probabilities that sum to 1."""

    e = np.exp(z - np.max(z))  # Subtract the maximum to prevent exp overflow

    return e / e.sum()


def forward_pass(layer_neuron, inputs, weights, biases):
    """
    Compute the network output one layer at a time.
    Return the final output and the activations from every layer.
    """

    activations = []  # activations[layer][neuron], after ReLU in hidden layers

    inputs = np.asarray(inputs, dtype=float)

    for layer_i in range(len(layer_neuron)):
        # Compute all neurons in the layer together: W @ inputs + bias
        # (np.asarray also accepts lists loaded from model.json)
        outputs_layer = np.asarray(weights[layer_i]) @ inputs + np.asarray(biases[layer_i])

        # Apply ReLU to every layer except the output layer: max(0, x)
        if layer_i != len(layer_neuron) - 1:
            outputs_layer = np.maximum(0, outputs_layer)
        else:
            # Apply Softmax to the output layer to produce probabilities
            outputs_layer = softmax(outputs_layer)

        activations.append(outputs_layer)
        inputs = outputs_layer  # This layer's outputs become the next layer's inputs

    # After the loop, `inputs` contains the output of the final layer
    return inputs, activations


# ============================================================
# Backpropagation
# ============================================================

def backward_pass(layer_neuron, outputs, activations, weights, target):
    """
    Compute the loss gradient for every neuron.
    - Output layer (Softmax + cross-entropy): the gradient simplifies to
      (output - target).
    - Hidden layers (ReLU): the gradient from the next layer is multiplied by
      the ReLU derivative (1 if the neuron was active, 0 otherwise).
    """

    gradients = []  # gradients[layer][neuron]

    # --- Output layer: derivative of (Softmax + cross-entropy) ---
    gradients.append(outputs - target)

    # --- Hidden layers: work backward from the penultimate layer to the first ---
    # A neuron's gradient is the weighted sum of the next layer's gradients,
    # multiplied by the derivative of ReLU.
    for layer_i in range(len(layer_neuron) - 2, -1, -1):
        previous_layer_gradients = (
            weights[layer_i + 1].T @ gradients[0]
            * (activations[layer_i] > 0)  # ReLU derivative
        )

        # Insert at the front to preserve layer order (input -> output)
        gradients.insert(0, previous_layer_gradients)

    return gradients


# ============================================================
# Parameter updates
# ============================================================

def update_parameters(layer_neuron, inputs, weights, biases, activations, gradients, lr):
    """Gradient descent: move each parameter in the direction that reduces the loss."""

    for layer_i in range(len(layer_neuron)):
        # Values entering this layer
        if layer_i == 0:
            previous_activations = inputs
        else:
            previous_activations = activations[layer_i - 1]

        # Weight gradient = neuron gradient * corresponding input value
        weights[layer_i] -= lr * np.outer(gradients[layer_i], previous_activations)

        # A bias acts like a weight whose input is always 1
        biases[layer_i] -= lr * gradients[layer_i]

    return weights, biases


# ============================================================
# Loss and evaluation
# ============================================================

def cross_entropy(outputs, target):
    """Compute cross-entropy between predicted probabilities and the target."""

    # 1e-12 prevents log(0) if a probability is exactly zero
    return -np.sum(target * np.log(outputs + 1e-12))


def accuracy(X, labels, layer_neuron, weights, biases):
    """Return the proportion of correctly classified examples (between 0 and 1)."""

    # Process all test examples together (one row per image)
    activations = np.asarray(X, dtype=float)

    for layer_i in range(len(layer_neuron)):
        activations = activations @ np.asarray(weights[layer_i]).T + np.asarray(biases[layer_i])

        if layer_i != len(layer_neuron) - 1:
            activations = np.maximum(0, activations)

    # The prediction is the digit with the highest score.
    # Softmax preserves the order, so it is not needed here.
    predictions = np.argmax(activations, axis=1)

    return float(np.mean(predictions == np.asarray(labels)))


# ============================================================
# Training
# ============================================================

def step(inputs, target, weights, biases, layer_neuron, lr):
    """
    Perform one parameter update for a single example.
    Return the loss measured before the update and the updated parameters.
    """

    # 1. Forward propagation
    outputs, activations = forward_pass(layer_neuron, inputs, weights, biases)

    # 2. Calculate the loss
    loss = cross_entropy(outputs, target)

    # 3. Backpropagation
    gradients = backward_pass(layer_neuron, outputs, activations, weights, target)

    # 4. Update parameters
    weights, biases = update_parameters(layer_neuron, inputs, weights, biases, activations, gradients, lr)

    return loss, weights, biases


def to_lists(weights, biases):
    """Convert NumPy parameters to Python lists (the model.json format)."""

    return [w.tolist() for w in weights], [b.tolist() for b in biases]


def train(X, Y, layer_neuron, epochs=EPOCHS, lr=LR, shuffle=True,
          test_data=None, on_epoch=None):
    """
    Train the network on the dataset (X: inputs, Y: targets).
    Each example is used once per epoch and triggers a parameter update
    (stochastic gradient descent). Shuffling examples between epochs prevents
    their order from influencing learning.
    - test_data: (X_test, labels_test), used to measure accuracy on unseen data
      after each epoch.
    - on_epoch: callback invoked after each epoch with (weights, biases), for
      example to save the model.
    Return the learned weights and biases.
    """

    if len(X) == 0 or len(X) != len(Y) or len(Y[0]) == 0:
        raise ValueError("X and Y must be non-empty and have the same length.")

    if len(layer_neuron) < 1:
        raise ValueError("The architecture must contain at least one layer.")

    if layer_neuron[-1] != len(Y[0]):
        raise ValueError("The last layer size must match the target output size.")

    # Convert to NumPy arrays once instead of repeating the conversion per example
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    weights, biases = generate_parameters(layer_neuron, X[0])
    order = list(range(len(X)))

    for epoch in range(epochs):
        start = time.time()

        if shuffle:
            random.shuffle(order)

        epoch_loss = 0

        for i in order:
            loss, weights, biases = step(X[i], Y[i], weights, biases, layer_neuron, lr)
            epoch_loss += loss

        epoch_loss /= len(order)

        # Print progress after each epoch so long runs provide feedback
        message = f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}"

        if test_data is not None:
            X_test, labels_test = test_data
            acc = accuracy(X_test, labels_test, layer_neuron, weights, biases)
            message += f" | Test accuracy: {acc * 100:.2f}%"

        message += f" | {time.time() - start:.0f} s"
        print(message, flush=True)

        # Save a checkpoint so progress is preserved if training stops
        if on_epoch is not None:
            on_epoch(*to_lists(weights, biases))

    return to_lists(weights, biases)