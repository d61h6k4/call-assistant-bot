import argparse
import json

from pathlib import Path


def create_gen_config(config):
    return {
        "model": {
            "type": config["model_type"],
            "pad_token_id": config["pad_token_id"],
            "eos_token_id": config["eos_token_id"],
            "bos_token_id": config["bos_token_id"],
            "decoder_start_token_id": config["text_config"]["decoder_start_token_id"],
            "vocab_size": config["vocab_size"],
            "context_length": config["projection_dim"],
            "encoder_decoder_init": {"filename": "encoder_model.onnx"},
            "embedding": {
                "filename": "embed_tokens.onnx",
                "inputs": {"input_ids": "input_ids"},
                # "outputs": {"embeddings": "inputs_embeds"},
            },
            "vision": {
                "filename": "vision_encoder_with_preprocessing.onnx",
                "inputs": {"pixel_values": "image"},
                "outputs": {
                    "visual_features": "image_features",
                },
            },
            "decoder": {
                "filename": "decoder_model_merged.onnx",
                "hidden_size": config["text_config"]["d_model"],
                "num_hidden_layers": config["text_config"]["num_hidden_layers"],
                "num_key_value_heads": config["text_config"]["decoder_attention_heads"],
                "head_size": 64,
                "inputs": {
                    "attention_mask": "encoder_attention_mask",
                    "past_key_names": "past_key_values.%d.decoder.key",
                    "past_value_names": "past_key_values.%d.decoder.value",
                    "cross_past_key_names": "past_key_values.%d.encoder.key",
                    "cross_past_value_names": "past_key_values.%d.encoder.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.decoder.key",
                    "present_value_names": "present.%d.decoder.value",
                    "cross_present_key_names": "present.%d.encoder.key",
                    "cross_present_value_names": "present.%d.encoder.value",
                },
            },
        },
        "search": {
            "do_sample": True,
            "early_stopping": True,
            "diversity_penalty": 0.0,
            "min_length": 0,
            "max_length": 50,
            "num_beams": 3,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "past_present_share_buffer": False,
            "num_return_sequences": 1,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_json",
        type=Path,
        help="Specify path to the model's config.json file",
        required=True,
    )
    parser.add_argument(
        "--genai_config_json",
        type=Path,
        help="Specify path to store generated genai_config.json",
        required=True,
    )
    return parser.parse_args()


def main(args):
    config = json.loads(args.config_json.read_text())
    args.genai_config_json.write_text(json.dumps(create_gen_config(config), indent=2))


if __name__ == "__main__":
    main(parse_args())
