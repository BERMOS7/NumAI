import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_NumAI.mnist import load_images, load_labels


class MnistLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)

    def _write_gzip(self, filename, content):
        path = self.data_dir / filename
        with gzip.open(path, "wb") as file:
            file.write(content)
        return path

    def test_load_images_flattens_and_normalizes_pixels(self):
        header = struct.pack(">IIII", 2051, 2, 2, 2)
        path = self._write_gzip("images.gz", header + bytes(range(8)))

        images = load_images(path)

        self.assertEqual(images.shape, (2, 4))
        np.testing.assert_allclose(images[0], [0, 1 / 255, 2 / 255, 3 / 255])
        self.assertAlmostEqual(float(images.max()), 7 / 255)

    def test_load_images_rejects_incomplete_pixel_data(self):
        header = struct.pack(">IIII", 2051, 2, 2, 2)
        path = self._write_gzip("incomplete-images.gz", header + b"\x00")

        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_images(path)

    def test_load_labels_reads_each_label(self):
        header = struct.pack(">II", 2049, 3)
        path = self._write_gzip("labels.gz", header + bytes([2, 5, 9]))

        self.assertEqual(load_labels(path), [2, 5, 9])

    def test_load_labels_rejects_wrong_data_length(self):
        header = struct.pack(">II", 2049, 3)
        path = self._write_gzip("incomplete-labels.gz", header + bytes([2, 5]))

        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_labels(path)


if __name__ == "__main__":
    unittest.main()
