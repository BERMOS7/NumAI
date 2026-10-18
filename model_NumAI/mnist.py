"""
Load images and labels from the MNIST dataset.
Images are normalized so they can be passed directly to the network.
"""

# ============================================================
# Imports
# ============================================================
import gzip
import struct
from pathlib import Path

import numpy as np

# Resolve data paths relative to the project, regardless of the current directory.
MNIST_DIR = Path(__file__).resolve().parent.parent / "MNIST"

# ============================================================
# Data loading
# ============================================================

def load_images(path):
    """Load MNIST images and convert them to normalized pixel vectors."""

    with gzip.open(path, "rb") as f:
        # IDX header fields are unsigned 4-byte integers in network (big-endian) order.
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))

        # The magic number 2051 identifies an MNIST image file.
        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")

        # Each pixel is an unsigned byte; frombuffer avoids element-by-element conversion.
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # Check that the file contains the number of pixels declared in the header.
        if len(data) != count * rows * cols:
            raise ValueError("Incomplete MNIST image data.")

    # Each row is a flattened image; division scales pixel values to the range [0, 1].
    return data.reshape(count, rows * cols) / 255.0


def load_labels(path):
    """Load the labels associated with MNIST images."""

    with gzip.open(path, "rb") as f:
        # The IDX header uses big-endian integers, just like the image header.
        magic, count = struct.unpack(">II", f.read(8))

        # The magic number 2049 identifies an MNIST label file.
        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")

        # Each label is stored in one byte and represents a digit from 0 to 9.
        data = f.read()

        # Also reject truncated files and files containing extra bytes.
        if len(data) != count:
            raise ValueError("Incomplete or invalid MNIST label data.")

        labels = list(data)

    return labels


# ============================================================
# Load training and test sets
# ============================================================

def load_mnist():
    """Load the images and labels in the training set."""

    images = load_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    labels = load_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")

    # Each image must have exactly one corresponding label.
    if len(images) != len(labels):
        raise ValueError("Image and label counts do not match.")

    return images, labels


def load_mnist_test():
    """Load the images and labels in the test set."""

    images = load_images(MNIST_DIR / "t10k-images-idx3-ubyte.gz")
    labels = load_labels(MNIST_DIR / "t10k-labels-idx1-ubyte.gz")

    # Each image must have exactly one corresponding label.
    if len(images) != len(labels):
        raise ValueError("Image and label counts do not match.")

    return images, labels