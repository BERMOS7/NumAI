"""Handle mouse drawing and its Pillow image copy."""

# ============================================================
# Imports and brush configuration
# ============================================================
from PIL import Image, ImageDraw

SIZE = 10  # Brush radius in pixels
CANVAS_SIZE = 500  # Matches the canvas dimensions in ui.py

# ============================================================
# Drawing state
# ============================================================

# Last mouse position; None means that no stroke is in progress
_last = {"x": None, "y": None}

# Grayscale image ("L"): 0 = black, 255 = white, as in the MNIST data
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
_pil = ImageDraw.Draw(image)


# ============================================================
# Drawing and clearing
# ============================================================

def reset_image():
    """Clear the Pillow image when the canvas drawing is cleared."""

    _pil.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)


def start_stroke(event, canvas, drawing_locked):
    """Start a stroke at the clicked point and remember its position."""

    if drawing_locked:
        return

    _last["x"], _last["y"] = event.x, event.y

    # Draw in Tkinter for display and in Pillow for the model input
    canvas.create_oval(
        event.x - SIZE,
        event.y - SIZE,
        event.x + SIZE,
        event.y + SIZE,
        fill="black",
        outline="black",
        tags=("ink",)
    )
    _pil.ellipse((event.x - SIZE, event.y - SIZE, event.x + SIZE, event.y + SIZE), fill=255)


def draw(event, canvas, drawing_locked):
    """Extend the stroke to the current mouse position."""

    if drawing_locked or _last["x"] is None:
        return

    # Rounded line caps and Pillow circles prevent gaps in the stroke
    canvas.create_line(
        _last["x"], _last["y"], event.x, event.y,
        width=2 * SIZE,
        capstyle="round",
        fill="black",
        tags=("ink",)
    )
    _pil.line((_last["x"], _last["y"], event.x, event.y), fill=255, width=2 * SIZE)
    _pil.ellipse((event.x - SIZE, event.y - SIZE, event.x + SIZE, event.y + SIZE), fill=255)

    _last["x"], _last["y"] = event.x, event.y


def end_stroke(event=None):
    """End the stroke so the next click starts a new segment."""

    _last["x"], _last["y"] = None, None