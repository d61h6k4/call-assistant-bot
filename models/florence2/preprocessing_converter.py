
import numpy as np
import onnx
from onnxruntime_extensions import OrtPyFunction
from onnxruntime_extensions.tools.pre_post_processing import (
    PrePostProcessor, create_named_value, ConvertImageToBGR, ReverseAxis,
    Resize, Normalize, ImageBytesToFloat, Unsqueeze, Transpose)


def image_processor():
    mean_std = list(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    steps = [
        ConvertImageToBGR(),
        ReverseAxis(),
        Transpose([2, 0, 1]),
        Resize((768, 768), layout="CHW"),
        ImageBytesToFloat(rescale_factor=0.00392156862745098),
        Normalize(mean_std, layout="CHW"),
        Unsqueeze([0]), # Add batch dim
    ]
    return steps

def convert(old_model_path: str, new_model_path: str):
    model = onnx.load(old_model_path)
    inputs = [
        create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])
    ]
    pipeline = PrePostProcessor(inputs, 18)

    preprocessing = image_processor()
    pipeline.add_pre_processing(preprocessing)
    pipeline._pre_processing_joins = [(preprocessing[-1], 0, "pixel_values")]

    new_model = pipeline.run(model)
    onnx.save_model(new_model, new_model_path)

if __name__ == "__main__":

    old_model_path = "models/florence2/data/vision_encoder.onnx"
    new_model_path = "/tmp/vision_encoder_with_postprocessing.onnx"
    convert(old_model_path, new_model_path)

    m = OrtPyFunction.from_model(new_model_path)

    x = np.fromfile("models/florence2/data/car.jpg", dtype=np.uint8)
    r = m(x)
    print(r)
    print(r.shape)
