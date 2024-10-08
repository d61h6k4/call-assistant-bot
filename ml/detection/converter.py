import argparse
import tempfile
import platform
from pathlib import Path

import onnx
from onnxruntime_extensions.tools.pre_post_processing.utils import IoMapEntry
from optimum.exporters.onnx.model_configs import DetrOnnxConfig
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoConfig
from onnxruntime_extensions.tools.pre_post_processing import (
    ChannelsLastToChannelsFirst,
    Identity,
    create_named_value,
    Resize,
    ImageBytesToFloat,
    Normalize,
    Unsqueeze,
    PrePostProcessor,
    Softmax,
    Squeeze,
    SelectBestBoundingBoxesByNMS,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_model",
        help="Specify path to the model to convert.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output_model",
        help="Specify path to store a converted model.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--quantization",
        help="Specify architecture or instruction set to quantize for",
        choices=["arm64", "avx2", "avx512", "avx512_vnni"],
        default="arm64" if platform.system() == "Darwin" else "avx512_vnni",
    )
    return parser.parse_args()


class ConditionalDetrOnnxConfig(DetrOnnxConfig):
    DEFAULT_ONNX_OPSET = 17

    def __init__(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config, task="object-detection")


def image_processor():
    steps = []

    steps.append(ChannelsLastToChannelsFirst())
    steps.append(Resize((504, 896), layout="CHW"))
    steps.append(ImageBytesToFloat(rescale_factor=0.00392156862745098))
    mean_std = list(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    steps.append(Normalize(mean_std, layout="CHW"))
    steps.append(Unsqueeze([0]))

    return steps


def object_detection_postprocessor():
    class Scores(Softmax):
        def __init__(self):
            super().__init__()
            self.input_names = ["logits"]
            self.output_names = ["probabilities"]

    class SqueezeLogits(Squeeze):
        def __init__(self):
            super().__init__(axes=[0], name="SqueezeLogits")
            self.input_names = ["probabilities"]
            self.output_names = ["scores"]

    class SqueezeBoxes(Squeeze):
        def __init__(self):
            super().__init__(axes=[0], name="SqueezeBoxes")
            self.input_names = ["pred_boxes"]
            self.output_names = ["boxes"]

    return [
        Identity(num_inputs=2),
        Scores(),
        SqueezeLogits(),
        (
            SqueezeBoxes(),
            [IoMapEntry(producer="Identity", producer_idx=1, consumer_idx=0)],
        ),
        (
            SelectBestBoundingBoxesByNMS(),
            [
                IoMapEntry(producer="SqueezeBoxes", producer_idx=0, consumer_idx=0),
                IoMapEntry(producer="SqueezeLogits", producer_idx=0, consumer_idx=1),
            ],
        ),
    ]


def convert(model_name: str, output_model: Path, quantization: str):
    custom_onnx_config = {"model": ConditionalDetrOnnxConfig(model_name)}

    with tempfile.TemporaryDirectory() as tmpdir:
        main_export(
            model_name,
            output=tmpdir,
            task="object-detection",
            custom_onnx_configs=custom_onnx_config,
        )

        onnx_model_body = tmpdir + "/model.onnx"
        model = onnx.load(onnx_model_body)
        inputs = [
            create_named_value("image", onnx.TensorProto.UINT8, ["height", "width", 3])
        ]

        pipeline = PrePostProcessor(
            inputs, custom_onnx_config["model"].DEFAULT_ONNX_OPSET
        )

        preprocessing = image_processor()
        pipeline.add_pre_processing(preprocessing)
        pipeline._pre_processing_joins = [(preprocessing[-1], 0, "pixel_values")]
        pipeline.add_post_processing(object_detection_postprocessor())
        model_with_preprocessing = pipeline.run(model)

        output_file = output_model / "model.onnx"
        onnx.save_model(model_with_preprocessing, output_file)

        if (output_model / "model_quantized.onnx").exists():
            (output_model / "model_quantized.onnx").unlink()

        quantizer = ORTQuantizer.from_pretrained(output_model)
        dqconfig = getattr(AutoQuantizationConfig, quantization)(
            is_static=False, per_channel=False
        )
        # quantization avx512* uses int8
        # onnxruntime doesn't have Conv_quant for int8 only for uint8
        if quantization in ("avx512", "avx512_vnni", "arm64"):
            dqconfig.nodes_to_exclude = ["Conv_quant"]
            dqconfig.operators_to_quantize = [
                "MatMul",
                "Attention",
                "LSTM",
                "Gather",
                "Transpose",
                "EmbedLayerNormalization",
            ]

        quantizer.quantize(save_dir=output_model, quantization_config=dqconfig)


def main():
    args = parse_args()
    model_name = str(args.input_model)
    convert(model_name, args.output_model, args.quantization)


if __name__ == "__main__":
    main()
