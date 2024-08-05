import easyocr
import unittest
import numpy as np
import cv2
import torch

from collections import OrderedDict
from pathlib import Path
from PIL import Image
from onnxruntime_extensions import OrtPyFunction


class OCRTest(unittest.TestCase):
    def setUp(self):
        self.model = easyocr.Reader(["ru", "en"], gpu=False)
        self.onnx_model = OrtPyFunction.from_model("ml/ocr/models/model.onnx")

    def test_sanity_check(self):
        for image_name in [
            "AlumniHub_bot.png",
            "AI-kit_Meeting_Bot.png",
            "Danila_Petrov.png",
            "Dina_Karakash.png",
            "Rinat_Kurbanov.png",
            "Denis_Semenov.png",
        ]:
            image = np.asarray(Image.open("testdata/" + image_name))
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # img = np.expand_dims(img_cv_grey, axis=-1)
            preds = self.onnx_model(img_cv_grey)
            preds_str = self.model.converter.decode_greedy(preds[0], [64])
            s = self.model.recognize(img_cv_grey, reformat=False)
            self.assertEqual(preds_str[0], s[0][1])


if __name__ == "__main__":
    unittest.main()
