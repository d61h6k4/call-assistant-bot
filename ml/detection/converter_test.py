from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
import platform
import numpy as np
import torch
from PIL import Image
import requests

from onnxruntime_extensions import OrtPyFunction
import unittest

from ml.detection.converter import convert
import tempfile
from pathlib import Path


class CDetrOnnxTest(unittest.TestCase):
    def setUp(self) -> None:
        url = "testdata/meeting_frame.png"
        self.image = Image.open(url)

        self.model_name = "microsoft/conditional-detr-resnet-50"

    def test_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_model = Path(tmpdir)
            convert(
                self.model_name,
                output_model,
                "avx512_vnni" if platform.system() == "Linux" else arm64,
            )

            processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                size={"shortest_edge": 504, "longest_edge": 896},
            )
            inputs = processor(images=self.image, return_tensors="pt")

            model = ConditionalDetrForObjectDetection.from_pretrained(self.model_name)
            with torch.no_grad():
                outputs = model(**inputs)

            onnx_model = OrtPyFunction.from_model(
                str(output_model / "model_quantized.onnx")
            )
            onnx_output = onnx_model(np.transpose(np.array(self.image), (2, 0, 1)))

            print(processor.post_process_object_detection(outputs))
            print(onnx_output)


if __name__ == "__main__":
    unittest.main()
