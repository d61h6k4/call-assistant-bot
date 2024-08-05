import argparse
import easyocr
import easyocr.model.vgg_model
import numpy as np
import onnx
import torch

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

from onnxruntime_extensions.tools.pre_post_processing import (
    ChannelsLastToChannelsFirst,
    Identity,
    create_named_value,
    Resize,
    ImageBytesToFloat,
    Normalize,
    Unsqueeze,
    PrePostProcessor,
    ArgMax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_model",
        help="Specify directory to store converted model's data.",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def image_processor(mean: tuple[float, ...], std: tuple[float, ...]):
    return [
        # ChannelsLastToChannelsFirst(),
        Resize((64, 256), layout="HW"),
        # create channel
        Unsqueeze([0]),
        ImageBytesToFloat(rescale_factor=0.00392156862745098),
        # Normalize(list(zip(mean, std)), layout="CHW"),
        # create batch
        Unsqueeze([0]),
    ]


def ocr_postprocessing():
    return [ArgMax()]


def convert(output_model: Path):
    config = easyocr.config.recognition_models["gen2"]["cyrillic_g2"]
    with TemporaryDirectory() as tmpdir:
        easyocr.utils.download_and_unzip(config["url"], config["filename"], tmpdir)

        recog_network = "generation2"
        network_params = {"input_channel": 1, "output_channel": 256, "hidden_size": 256}
        recognizer = easyocr.model.vgg_model.Model(
            # +1 is blank for CTCLoss
            num_class=len(config["characters"]) + 1,
            **network_params,
        )
        state_dict = torch.load(
            tmpdir + f"/{config['filename']}", map_location="cpu", weights_only=False
        )
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        recognizer.load_state_dict(new_state_dict)

        batch_size_1_1 = 256
        in_shape_1 = [1, 1, 64, batch_size_1_1]
        dummy_input_1 = torch.rand(in_shape_1)
        dummy_input_1 = dummy_input_1

        batch_size_2_1 = 25
        in_shape_2 = [1, batch_size_2_1]
        dummy_input_2 = torch.rand(in_shape_2)
        dummy_input_2 = dummy_input_2

        dummy_input = (dummy_input_1, dummy_input_2)

        torch.onnx.export(
            recognizer,
            dummy_input,
            tmpdir + "/model.onnx",
            export_params=True,
            opset_version=18,
            input_names=["input", "input2"],
            output_names=["output"],
            verbose=True,
        )

        model = onnx.load(tmpdir + "/model.onnx")

        inputs = [
            create_named_value("image", onnx.TensorProto.UINT8, ["height", "width"])
        ]
        pipeline = PrePostProcessor(inputs, 18)
        preprocessing = image_processor([0.5], [0.5])
        pipeline.add_pre_processing(preprocessing)
        pipeline._pre_processing_joins = [(preprocessing[-1], 0, "input")]
        pipeline.add_post_processing(ocr_postprocessing())
        model_with_preprocessing = pipeline.run(model)
        onnx.save_model(model_with_preprocessing, str(output_model / "model.onnx"))


def main(args: argparse.Namespace):
    convert(args.output_model)


if __name__ == "__main__":
    main(parse_args())
