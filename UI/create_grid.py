"""Extract a pixel grid from the drawing on the canvas."""

# ============================================================
# Grid extraction
# ============================================================


def create_grid(canvas, size=28):
    """
    Convert the canvas contents into a binary grid of the given `size`.
    Each cell is 1 if it overlaps a stroke, and 0 otherwise.
    """

    if size <= 0:
        raise ValueError("Grid size must be positive.")

    # The canvas size is only valid after Tkinter has initialized it
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    if width <= 1 or height <= 1:
        raise ValueError("The canvas has not been initialized yet.")

    grid = []
    cell_width = width / size
    cell_height = height / size

    # Check whether each cell overlaps an object tagged as ink
    for row in range(size):
        line = []

        for col in range(size):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height

            objects = canvas.find_overlapping(x1, y1, x2, y2)

            # Tkinter returns every object overlapping the cell; the "ink" tag
            # excludes grid lines and other canvas items.
            ink_found = any(
                "ink" in canvas.gettags(obj)
                for obj in objects
            )

            line.append(1.0 if ink_found else 0.0)

        grid.append(line)

    # Draw the grid over the image after extracting its pixels
    for i in range(size + 1):
        x = i * cell_width
        y = i * cell_height

        canvas.create_line(
            x, 0, x, height,
            fill="#cccccc",
            tags=("grid",)
        )
        canvas.create_line(
            0, y, width, y,
            fill="#cccccc",
            tags=("grid",)
        )

    return grid
