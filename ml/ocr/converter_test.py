import unittest
import numpy as np

from pathlib import Path
from PIL import Image
from onnxruntime_extensions import OrtPyFunction


class OCRTest(unittest.TestCase):
    def setUp(self):
        self.model = OrtPyFunction.from_model("ml/ocr/models/model.onnx")
        self.vocab = {
            int(row[1]): row[0]
            for row in map(
                lambda row: row.split("\t"),
                Path("ml/ocr/models/vocab.tsv").read_text().split("\n"),
            )
        }

    def test_recognition(self):
        for image_name in [
            "AlumniHub_bot.png",
            "AI-kit_Meeting_Bot.png",
            "Danila_Petrov.png",
            "Dina_Karakash.png",
            "Rinat_Kurbanov.png",
            "Denis_Semenov.png",
        ]:
            image = Image.open("testdata/" + image_name)
            input_tensor = np.asarray(image)
            out = self.model(input_tensor)
            s = "".join(self.vocab.get(ix, "<UNK>") for ix in out[0])

            print(f"{image_name} -> {s}")


if __name__ == "__main__":
    unittest.main()
