"""Build and manage the NumAI graphical user interface."""

# ============================================================
# Imports
# ============================================================
import tkinter as tk

from UI.drawing import draw, start_stroke, end_stroke, reset_image, image
from UI.prepare_drawing import prepare_drawing
from model_NumAI.predict import predict


# ============================================================
# Interface creation
# ============================================================

def launch():
    """Create the drawing window and start its event loop."""

    # Prevent drawing changes after a prediction until the canvas is cleared.
    drawing_locked = False

    window = tk.Tk()
    window.title("NumAI")
    window.geometry("600x650")
    window.resizable(False, False)
    window.configure(bg="#202124")

    # Application title
    title = tk.Label(
        window,
        text="NumAI",
        font=("Arial", 24, "bold"),
        bg="#202124",
        fg="white"
    )
    title.pack(pady=5)

    # Drawing area; its dimensions match the Pillow image
    canvas = tk.Canvas(
        window,
        bg="white",
        width=500,
        height=500,
        highlightthickness=2,
        highlightbackground="#555555"
    )
    canvas.pack()

    # Bottom area containing the result and action buttons
    bottom_frame = tk.Frame(window, bg="#202124")
    bottom_frame.pack(fill="x", pady=8)

    result_label = tk.Label(
        bottom_frame,
        text="Result: —",
        font=("Arial", 18, "bold"),
        bg="#202124",
        fg="white"
    )
    result_label.pack(side="left", padx=20)

    button_frame = tk.Frame(bottom_frame, bg="#202124")
    button_frame.pack(side="right", padx=20)

    # ========================================================
    # Button actions
    # ========================================================

    def clear():
        """Clear the canvas and its image copy, then reset the result."""

        # drawing_locked is shared with the callbacks defined in launch.
        nonlocal drawing_locked

        canvas.delete("all")
        reset_image()  # Also clear the Pillow copy of the drawing
        end_stroke()
        drawing_locked = False
        result_label.config(text="Result: —")

    def validate():
        """Preprocess the drawing, request a prediction, and display the result."""

        # The drawing functions read the same variable captured by their callbacks.
        nonlocal drawing_locked

        try:
            # Get the drawing and preprocess it like an MNIST image
            inputs = prepare_drawing(image)

            # Get the predicted digit and its confidence from the neural network
            digit, confidence, _ = predict(inputs)

            # Lock the drawing only after a successful prediction.
            drawing_locked = True
            result_label.config(
                text=f"Prediction: {digit} (confidence: {confidence * 100:.0f}%)"
            )

        except (ValueError, FileNotFoundError, KeyError) as error:
            result_label.config(text="Error: model or drawing")
            print("Prediction error:", error)

    # ========================================================
    # Buttons and drawing events
    # ========================================================

    clear_button = tk.Button(
        button_frame,
        text="Clear",
        command=clear,
        font=("Arial", 10, "bold"),
        width=9,
        bg="#333333",
        fg="white",
        activebackground="#444444",
        activeforeground="white",
        relief="flat"
    )

    validate_button = tk.Button(
        button_frame,
        text="Predict",
        command=validate,
        font=("Arial", 10, "bold"),
        width=9,
        bg="#4a90e2",
        fg="white",
        activebackground="#357abd",
        activeforeground="white",
        relief="flat"
    )

    clear_button.pack(side="left", padx=4)
    validate_button.pack(side="left", padx=4)

    # Bind mouse gestures to stroke start, continuation, and end
    canvas.bind("<Button-1>", lambda e: start_stroke(e, canvas, drawing_locked))
    canvas.bind("<B1-Motion>", lambda e: draw(e, canvas, drawing_locked))
    canvas.bind("<ButtonRelease-1>", end_stroke)

    # Keep the window open and process Tkinter events
    window.mainloop()