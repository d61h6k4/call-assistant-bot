from ml.detection.ml_server import Model
import unittest
import json
from PIL import Image


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Model()
        self.image = Image.open("testdata/meeting_frame.png")

    def test_sanity_check(self):
        print(json.dumps({"result": self.model("test_image", self.image)}))


if __name__ == "__main__":
    unittest.main()
