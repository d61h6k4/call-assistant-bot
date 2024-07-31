import argparse
import numpy as np
import onnx

from pathlib import Path
from onnxruntime_extensions import OrtPyFunction
from onnxruntime_extensions.tools.pre_post_processing import (
    PrePostProcessor,
    create_named_value,
    ConvertImageToBGR,
    ReverseAxis,
    Resize,
    Normalize,
    ImageBytesToFloat,
    Unsqueeze,
    Transpose,
    ChannelsLastToChannelsFirst,
)


def image_processor():
    mean_std = list(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    steps = [
        ChannelsLastToChannelsFirst(),
        Resize((768, 768), layout="CHW"),
        ImageBytesToFloat(rescale_factor=0.00392156862745098),
        Normalize(mean_std, layout="CHW"),
        Unsqueeze([0]),  # Add batch dim
    ]
    return steps


def convert(old_model_path: str, new_model_path: str):
    model = onnx.load(old_model_path)
    inputs = [
        create_named_value("image", onnx.TensorProto.UINT8, ["height", "width", 3])
    ]
    pipeline = PrePostProcessor(inputs, 18)

    preprocessing = image_processor()
    pipeline.add_pre_processing(preprocessing)
    pipeline._pre_processing_joins = [(preprocessing[-1], 0, "pixel_values")]

    new_model = pipeline.run(model)
    onnx.save_model(new_model, new_model_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_vision_encoder",
        type=Path,
        help="Specify path to the original Florence2 vision encoder model",
        required=True,
    )
    parser.add_argument(
        "--vision_encoder_with_preprocessing",
        type=Path,
        help="Specify path to store created vision encoder model with preprocessing steps",
        required=True,
    )
    return parser.parse_args()


def main(args):
    convert(args.original_vision_encoder, args.vision_encoder_with_preprocessing)


if __name__ == "__main__":
    main(parse_args())
