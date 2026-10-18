import unittest

from model_NumAI.predict import predict


class PredictTests(unittest.TestCase):
    def test_input_must_contain_784_pixels(self):
        with self.assertRaisesRegex(ValueError, "784 pixels"):
            predict([0.0] * 783)

    def test_prediction_returns_digit_and_probability_distribution(self):
        digit, confidence, probabilities = predict([0.0] * 784)

        self.assertIn(digit, range(10))
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertAlmostEqual(confidence, probabilities[digit])


if __name__ == "__main__":
    unittest.main()
