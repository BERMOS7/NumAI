"""Preprocess a user drawing before passing it to the network."""

# ============================================================
# Imports
# ============================================================
import numpy as np
from PIL import Image


# ============================================================
# Drawing preprocessing
# ============================================================

def prepare_drawing(img):
    """
    Convert a drawing (a Pillow image, white on black) into 784 values as in
    MNIST: fit the digit inside a 20x20 box and center it by its center of mass.
    """

    # 1. Crop the image to the bounding box of the digit
    box = img.getbbox()
    if box is None:
        raise ValueError("The drawing is empty.")
    digit = img.crop(box)

    # 2. Fit the digit in a 20-pixel box while preserving its aspect ratio.
    # LANCZOS interpolation smooths edges during resizing.
    width, height = digit.size
    scale = 20 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    # 3. Place the digit in the center of a black 28x28 image
    canvas28 = Image.new("L", (28, 28), 0)
    canvas28.paste(digit, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    # 4. The center of mass weights each pixel by its intensity, locating the
    #    visual center of the stroke rather than just the middle of its box.
    pixels = np.asarray(canvas28, dtype=float) / 255.0
    total = pixels.sum()
    center_y = (pixels.sum(axis=1) * np.arange(28)).sum() / total
    center_x = (pixels.sum(axis=0) * np.arange(28)).sum() / total

    # Shift pixels without wrapping them around to the opposite image edge
    shift_y = round(14 - center_y)
    shift_x = round(14 - center_x)
    shifted = np.zeros_like(pixels)

    source_y = slice(max(0, -shift_y), min(28, 28 - shift_y))
    source_x = slice(max(0, -shift_x), min(28, 28 - shift_x))
    target_y = slice(max(0, shift_y), min(28, 28 + shift_y))
    target_x = slice(max(0, shift_x), min(28, 28 + shift_x))
    shifted[target_y, target_x] = pixels[source_y, source_x]
    pixels = shifted

    # Flatten the matrix row by row into the network's 784-value input vector
    return pixels.flatten().tolist()