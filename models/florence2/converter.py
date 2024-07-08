
import os
from onnx import helper, numpy_helper, TensorProto, external_data_helper, save_model
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import torch


from onnxruntime_genai.models.builder import Model


# Trick to load florence wo flash_attn
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


class Falcon2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Context length defined as a max_position_embeddings
        # Used mostly for ROPE
        config.max_position_embeddings = config.text_config.max_position_embeddings
        # No idea why it's needed
        config.intermediate_size = -1
        config.hidden_size = config.text_config.d_model
        config.num_attention_heads = 12
        config.num_hidden_layers = -1
        config.hidden_act = -1
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

@patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports)
def convert():
    model_name = "microsoft/Florence-2-base-ft"
    output_dir = "models/florence2/data"
    precision = "fp32"
    execution_provider = "cpu"
    cache_dir = "cache_dir"

    # Create cache and output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model config
    extra_kwargs = {"cache_dir": cache_dir}
    hf_name = model_name
    config = AutoConfig.from_pretrained(hf_name, use_auth_token=True, trust_remote_code=True, **extra_kwargs)
    # Set input/output precision of ONNX model
    io_dtype = TensorProto.FLOAT if precision in {"int8", "fp32"} or (precision == "int4" and execution_provider == "cpu") else TensorProto.FLOAT16

    onnx_model = Falcon2Model(config, io_dtype, precision, execution_provider, cache_dir, {})

    extra_kwargs = {} if os.path.exists(onnx_model.model_name_or_path) else {"num_hidden_layers": onnx_model.num_layers} if "num_hidden_layers" in onnx_model.extra_options else {"cache_dir": onnx_model.cache_dir}
    print(extra_kwargs)

    model = AutoModelForCausalLM.from_pretrained(onnx_model.model_name_or_path, use_auth_token=True, trust_remote_code=True, **extra_kwargs)
    for module in model.modules():
        print(module.__class__.__name__)

    # Make ONNX model
    # onnx_model.make_model("")
    # Save ONNX model
    # onnx_model.save_model(output_dir)


if __name__ == "__main__":
    convert()
