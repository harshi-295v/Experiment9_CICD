
# Name: GLV Harshitha
# Registration Number: 231FA04F53

import os
import unittest
import joblib
import numpy as np

class TestModel(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("model.joblib"))

    def test_model_loads(self):
        model = joblib.load("model.joblib")
        self.assertIsNotNone(model)

    def test_model_prediction(self):
        model = joblib.load("model.joblib")

        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(sample)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1, 2])

if __name__ == "__main__":
    unittest.main()