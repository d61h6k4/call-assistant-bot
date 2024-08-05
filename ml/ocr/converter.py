import argparse
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from onnxtr.models.recognition.models.vitstr import default_cfgs
from onnxtr.models.engine import EngineConfig
from onnxtr.utils.data import download_from_url
from onnxtr.utils import VOCABS

import onnx
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
        ChannelsLastToChannelsFirst(),
        Resize((32, 128), layout="CHW"),
        ImageBytesToFloat(rescale_factor=0.00392156862745098),
        Normalize(list(zip(mean, std)), layout="CHW"),
        Unsqueeze([0]),
    ]


def ocr_postprocessing():
    return [ArgMax()]


def convert(cfg: EngineConfig, output_model: Path):
    vocab = cfg["vocab"]
    (output_model / "vocab.tsv").write_text(
        "\n".join("\t".join([ch, str(ix)]) for ix, ch in enumerate(vocab))
    )

    with TemporaryDirectory() as tmpdir:
        download_from_url(cfg["url_8_bit"], file_name="model.onnx", cache_dir=tmpdir)
        model = onnx.load(tmpdir + "/model.onnx")

        inputs = [
            create_named_value("image", onnx.TensorProto.UINT8, ["height", "width", 3])
        ]
        pipeline = PrePostProcessor(inputs, 18)
        preprocessing = image_processor(cfg["mean"], cfg["std"])
        pipeline.add_pre_processing(preprocessing)
        pipeline._pre_processing_joins = [(preprocessing[-1], 0, "input")]
        pipeline.add_post_processing(ocr_postprocessing())
        model_with_preprocessing = pipeline.run(model)
        onnx.save_model(model_with_preprocessing, str(output_model / "model.onnx"))


def main(args: argparse.Namespace):
    cfg = default_cfgs["vitstr_small"]
    convert(cfg, args.output_model)


if __name__ == "__main__":
    main(parse_args())
