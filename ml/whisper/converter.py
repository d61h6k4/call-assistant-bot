"""
Convert Whisper model to ONNX format.
"""
import os
import argparse
import tempfile
import subprocess

import onnx
from transformers import WhisperProcessor
from onnxruntime_extensions import OrtPyFunction, util
from onnxruntime_extensions.cvt import gen_processing_models


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert Whisper model to ONNX format")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "ml/whisper/models/onnx"),
                        help="Directory to save ONNX model")
    parser.add_argument("--precision", type=str, choices=["fp32", "int8"], default="fp32", help="Model precision")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization (only for int8)")
    return parser.parse_args()


def convert_whisper_to_onnx(
    model_name: str,
    cache_dir: str,
    output_dir: str,
    precision: str = "fp32",
    quantize: bool = False
):
    """
    Convert Whisper model to ONNX format.

    Args:
        model_name (str): Model name or path.
        output_dir (str): Directory to save ONNX model.
        precision (str, optional): Model precision. Default is "float32".
        quantize (bool, optional): Apply quantization. Default is False.
    """
    sub_args = [
        "python", '-m',
        'onnxruntime.transformers.models.whisper.convert_to_onnx',
        "--model_name_or_path", model_name,
        "--cache_dir", cache_dir,
        "--output", output_dir,
        "--precision", precision,
        "--use_external_data_format",
        "-r", "cpu"
    ]

    if precision == "int8" and quantize:
        sub_args.extend([
            "--quantize_embedding_layer",
        ])

    try:
        result = subprocess.run(sub_args, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"model saved to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during model conversion: {e}")
        print(e.stderr)
        return False


def process_test_file(model_name: str):
    """
    Return pre and post processing models.

    Args:
        model_name (str): Model name or path.

    Returns:
        tuple: Pre and post processing models.
    """

    _processor = WhisperProcessor.from_pretrained(model_name)
    pre_m, post_m = gen_processing_models(_processor,
                                          pre_kwargs={"USE_AUDIO_DECODER": True, "USE_ONNX_STFT": True},
                                          post_kwargs={},
                                          opset=17)
    return pre_m, post_m

def main():
    """Main function"""
    args = parse_args()
    with tempfile.TemporaryDirectory() as temp_dir:

        cache_dir = os.path.join(temp_dir, 'cache')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        result = convert_whisper_to_onnx(args.model_name, cache_dir, output_dir, args.precision, args.quantize)
        if not result:
            return
        pre_m, post_m = process_test_file(args.model_name)
        fn_core = OrtPyFunction.from_model(
            os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_beamsearch.onnx"),
            cpu_only=True
        )
        print('ready to merge')
        final_m = util.quick_merge(pre_m, fn_core.onnx_model, post_m)
        onnx.save(
            final_m, os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_e2e.onnx"),
            save_as_external_data=True
        )
    print(f"model saved to {args.output_dir}")

    return True

if __name__ == "__main__":
    main()
