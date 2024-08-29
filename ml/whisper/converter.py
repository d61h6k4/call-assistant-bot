"""
Convert Whisper model to ONNX format.
"""
import os
import shutil
import copy
import argparse
import tempfile
import logging
import torch

import onnx
from transformers import WhisperProcessor
from onnxruntime_extensions import OrtPyFunction, util
from onnxruntime_extensions.cvt import gen_processing_models

from onnxruntime.transformers.benchmark_helper import Precision, create_onnxruntime_session, prepare_environment
from onnxruntime.transformers.models.whisper.whisper_chain import chain_model
from onnxruntime.transformers.models.whisper.whisper_helper import WhisperHelper

from onnxruntime import quantization


logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert Whisper model to ONNX format")
    parser.add_argument("-m", "--model_name_or_path",
                        type=str, default="openai/whisper-large-v2", help="Model name or path")
    parser.add_argument("--final_dir", type=str, default=os.path.join(os.getcwd(), "ml/whisper/models/onnx"),
                        help="Directory to save ONNX model")
    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. FLOAT32 for full precision, FLOAT16 for half precision, INT8 for quantization",
    )
    parser.add_argument("--quantize", action="store_true", help="Apply quantization (only for int8)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.set_defaults(use_vocab_mask=False)
    parser.set_defaults(use_prefix_vocab_mask=False)
    parser.set_defaults(use_logits_processor=True)
    parser.set_defaults(use_int32_inputs=True)
    parser.set_defaults(quantize_embedding_layer=False)
    parser.set_defaults(quantize_per_channel=False)
    parser.set_defaults(quantize_reduce_range=False)
    parser.set_defaults(disable_auto_mixed_precision=False)
    parser.set_defaults(optimize_onnx=False)
    parser.set_defaults(use_forced_decoder_ids=False)
    parser.set_defaults(merge_encoder_and_decoder_init=True)
    parser.set_defaults(overwrite=True)
    parser.set_defaults(use_gpu=False)
    parser.set_defaults(use_external_data_format=True)
    parser.set_defaults(provider="CPUExecutionProvider")
    parser.set_defaults(collect_cross_qk=False)
    parser.set_defaults(extra_decoding_ids=False)
    parser.set_defaults(use_temperature=False)
    parser.set_defaults(output_sequence_scores=False)
    parser.set_defaults(output_scores=False)
    parser.set_defaults(output_no_speech_probs=False)
    parser.set_defaults(no_repeat_ngram_size=False)
    parser.set_defaults(output_cross_qk=False)
    parser.set_defaults(cross_qk_onnx_model=False)
    return parser.parse_args()


def export_onnx_models(
    model_name_or_path,
    model_impl,
    cache_dir,
    output_dir,
    use_gpu,
    use_external_data_format,
    optimize_onnx,
    precision,
    verbose,
    merge_encoder_and_decoder_init: bool = True,
    overwrite: bool = False,
    disable_auto_mixed_precision: bool = False,
    use_int32_inputs: bool = True,
    quantize_embedding_layer: bool = False,
    quantize_per_channel: bool = False,
    quantize_reduce_range: bool = False,
    state_dict_path: str = "",
    provider: str = "cpu",
):
    """
    Export ONNX models.
    """
    device = torch.device("cuda:0" if use_gpu else "cpu")

    models = WhisperHelper.load_model(
        model_name_or_path, model_impl, cache_dir, device, merge_encoder_and_decoder_init, state_dict_path
    )
    config = models["decoder"].config

    if (not use_external_data_format) and (config.num_hidden_layers > 24):
        logger.info("Try use_external_data_format when model size > 2GB")

    output_paths = []
    for name, model in models.items():
        print(f"========> Handling {name} model......")
        model.to(device)
        filename_suffix = "_" + name

        onnx_path = WhisperHelper.get_onnx_path(
            output_dir,
            model_name_or_path,
            suffix=filename_suffix,
            new_folder=False,
        )

        if overwrite or not os.path.exists(onnx_path):
            logger.info("Exporting ONNX model to %s", onnx_path)
            # We have to clone model before exporting onnx, otherwise verify_onnx will report large difference.
            device_to_export = torch.device("cpu")
            cloned_model = copy.deepcopy(model).to(device_to_export)
            WhisperHelper.export_onnx(
                cloned_model,
                device_to_export,
                onnx_path,
                verbose,
                use_external_data_format,
                use_int32_inputs=use_int32_inputs,
            )
        else:
            logger.info("Skip exporting: existed ONNX model %s", onnx_path)

        # Optimize ONNX graph. Note that we have not implemented graph optimization for Whisper yet.
        if optimize_onnx or precision != Precision.FLOAT32:
            output_path = WhisperHelper.get_onnx_path(
                output_dir,
                model_name_or_path,
                suffix=filename_suffix + "_" + str(precision),
                new_folder=False,
            )

            if overwrite or not os.path.exists(output_path):
                if optimize_onnx:
                    logger.info("Optimizing model to %s", output_path)
                    WhisperHelper.optimize_onnx(
                        onnx_path,
                        output_path,
                        precision == Precision.FLOAT16,
                        config.encoder_attention_heads,
                        config.d_model,
                        use_external_data_format,
                        auto_mixed_precision=not disable_auto_mixed_precision,
                        use_gpu=use_gpu,
                        provider=provider,
                    )
                    onnx_path = output_path

                if precision == Precision.INT8:
                    quantization.quantize_dynamic(
                        onnx_path,
                        output_path,
                        op_types_to_quantize=(
                            ["MatMul", "Gemm", "Gather"] if quantize_embedding_layer else ["MatMul", "Gemm"]
                        ),
                        use_external_data_format=use_external_data_format,
                        per_channel=quantize_per_channel,
                        reduce_range=quantize_reduce_range,
                        extra_options={"MatMulConstBOnly": True},
                    )
            else:
                logger.info("Skip optimizing: existing ONNX model %s", onnx_path)
        else:
            output_path = onnx_path

        ort_session = create_onnxruntime_session(
            output_path,
            use_gpu=use_gpu,
            provider=provider,
        )
        assert ort_session is not None

        output_paths.append(output_path)

    return output_paths

def convert_whisper_to_onnx(args):
    """
    Convert Whisper model to ONNX format.
    Args:
        model_name (str): model name or path.
        cache_dir (str): cache directory.
        output_dir (str): output directory.
        precision (str, optional): model precision. Defaults to "fp32".
        quantize (bool, optional): apply quantization. Defaults to False.

    Returns:
        List[str]: list of paths to saved ONNX models.
    """

    try:
        output_paths = export_onnx_models(
            args.model_name_or_path,
            model_impl="hf",
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
            use_external_data_format=args.use_external_data_format,
            optimize_onnx=args.optimize_onnx,
            precision=args.precision,
            verbose=args.verbose,
            merge_encoder_and_decoder_init=args.merge_encoder_and_decoder_init,
            overwrite=args.overwrite,
            disable_auto_mixed_precision=args.disable_auto_mixed_precision,
            use_int32_inputs=args.use_int32_inputs,
            quantize_embedding_layer=args.quantize_embedding_layer,
            quantize_per_channel=args.quantize_per_channel,
            quantize_reduce_range=args.quantize_reduce_range,
            provider=args.provider,
        )

        args.beam_model_output_dir = WhisperHelper.get_onnx_path(
            args.output_dir,
            args.model_name_or_path,
            suffix="_beamsearch",
            new_folder=False,
        )
        for path in output_paths:
            if "encoder_decoder" in path:
                args.encoder_path = path
            elif "decoder" in path:
                args.decoder_path = path
        chain_model(args)
        output_paths.append(args.beam_model_output_dir)
        ort_session = create_onnxruntime_session(
            args.beam_model_output_dir,
            use_gpu=args.use_gpu,
            provider=args.provider,
        )
        device = torch.device("cuda:0" if args.use_gpu else "cpu")
        max_diff = WhisperHelper.verify_onnx(args.model_name_or_path, args.cache_dir, ort_session, device)
        if max_diff > 1e-4:
            print("PyTorch and ONNX Runtime results are NOT close")
        else:
            print("PyTorch and ONNX Runtime results are close")
        return args.beam_model_output_dir
    except Exception as e:
        print(f"Error during model conversion: {e}")
        return None


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
        args.cache_dir = os.path.join(temp_dir, 'cache')
        args.output_dir = os.path.join(temp_dir, 'output')
        prepare_environment(args.cache_dir, args.output_dir, False)

        result = convert_whisper_to_onnx(args)
        if result is None:
            return
        pre_m, post_m = process_test_file(args.model_name_or_path)
        fn_core = OrtPyFunction.from_model(result, cpu_only=True)
        print('ready to merge')
        final_m = util.quick_merge(pre_m, fn_core.onnx_model, post_m)
        if os.path.exists(args.final_dir):
            shutil.rmtree(args.final_dir)
        os.makedirs(args.final_dir, exist_ok=True)
        onnx.save(
            final_m, os.path.join(args.final_dir, f"{args.model_name_or_path.split('/')[-1]}_fp32_e2e.onnx"),
            save_as_external_data=True
    )
    print(f"model saved to {args.output_dir}")

    return True

if __name__ == "__main__":
    main()
