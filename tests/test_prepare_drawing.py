import unittest

import numpy as np
from PIL import Image, ImageDraw

from UI.prepare_drawing import prepare_drawing


class PrepareDrawingTests(unittest.TestCase):
    def test_empty_image_is_rejected(self):
        image = Image.new("L", (500, 500), 0)

        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_drawing(image)

    def test_drawing_becomes_784_normalized_pixels(self):
        image = Image.new("L", (500, 500), 0)
        drawing = ImageDraw.Draw(image)
        drawing.line((100, 100, 300, 350), fill=255, width=20)

        pixels = prepare_drawing(image)

        self.assertEqual(len(pixels), 28 * 28)
        self.assertGreater(max(pixels), 0)
        self.assertGreaterEqual(min(pixels), 0)
        self.assertLessEqual(max(pixels), 1)

    def test_center_of_mass_is_near_image_center(self):
        image = Image.new("L", (120, 120), 0)
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((10, 20, 60, 90), fill=255)

        pixels = np.asarray(prepare_drawing(image)).reshape(28, 28)
        total = pixels.sum()
        center_y = (pixels.sum(axis=1) * np.arange(28)).sum() / total
        center_x = (pixels.sum(axis=0) * np.arange(28)).sum() / total

        self.assertAlmostEqual(center_x, 14, delta=1)
        self.assertAlmostEqual(center_y, 14, delta=1)


if __name__ == "__main__":
    unittest.main()
