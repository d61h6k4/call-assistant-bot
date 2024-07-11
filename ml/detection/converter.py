
import argparse
from pathlib import Path

from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
import torch
from PIL import Image
import requests
from optimum.exporters.onnx.model_configs import DetrOnnxConfig
from optimum.exporters.onnx import main_export
from transformers import AutoConfig
from transformers.onnx.config import DEFAULT_ONNX_OPSET


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model", help="Specify path to the model to convert.", required=True, type=Path)
    parser.add_argument("--output_model", help="Specify path to store a converted model.", required=True, type=Path)
    return parser.parse_args()

class ConditionalDetrOnnxConfig(DetrOnnxConfig):
    def __init__(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config, task="object-detection")



def main():
    args = parse_args()
    model_name = str(args.input_model)

    custom_onnx_config = {"model": ConditionalDetrOnnxConfig(model_name)}
    main_export(model_name, output=args.output_model, task="object-detection", custom_onnx_configs=custom_onnx_config)


if __name__ == "__main__":
    main()
