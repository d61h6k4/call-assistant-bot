from ml.detection.ml_server import Model
import unittest
import httpx
from PIL import Image


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Model()
        self.image = Image.open(
            httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg")
        )

    def test_sanity_check(self):
        print(self.model(self.image))


if __name__ == "__main__":
    unittest.main()
