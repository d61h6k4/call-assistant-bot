
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
import numpy as np
import torch
from PIL import Image
import requests

from onnxruntime_extensions import OrtPyFunction
import unittest

class CDetrOnnxTest(unittest.TestCase):
    def setUp(self) -> None:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def test_compare(self):
        processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")
        inputs = processor(images=self.image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        onnx_model = OrtPyFunction.from_model("/tmp/detr_onnx/model.onnx")
        onnx_output = onnx_model(inputs["pixel_values"].numpy())
        np.testing.assert_allclose(outputs["logits"].numpy(), onnx_output[0], atol=2e-4, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
