"""
Test for Whisper converter
"""
import unittest
import difflib
from onnxruntime_extensions import OrtPyFunction
import numpy as np


class WhisperConverter(unittest.TestCase):
    """
    Test for Whisper converter
    """

    def setUp(self):
        self.e2e_model = OrtPyFunction.from_model("ml/whisper/models/onnx/whisper-large-v2_fp32_e2e.onnx",
                                                  cpu_only=True)
        self.e2e_model_ort_session = self.e2e_model._ensure_ort_session() # pylint: disable=protected-access

        self.raw_audio = np.fromfile("testdata/whisper_audio.wav", dtype=np.uint8)
        self.raw_audio = np.expand_dims(self.raw_audio, axis=0)

    def _get_model_inputs(self, ort_session: OrtPyFunction, audio_data: np.ndarray):
        ort_names = list(map(lambda entry: entry.name, ort_session.get_inputs()))

        inputs = [
            audio_data,                          # audio_stream/input_features
            np.asarray([300], dtype=np.int32),   # max_length
            np.asarray([0], dtype=np.int32),     # min_length
            np.asarray([5], dtype=np.int32),     # num_beams
            np.asarray([1], dtype=np.int32),     # num_return_sequences
            np.asarray([0.95], dtype=np.float32), # length_penalty
            np.asarray([1.0], dtype=np.float32), # repetition_penalty
        ]
        required_input_names = {"audio_stream", "input_features", "max_length", "min_length", "num_beams",
                                "num_return_sequences", "length_penalty", "repetition_penalty"}
        batch_size = 1
        N_MELS = 128 # pylint: disable=invalid-name
        N_FRAMES = 3000 # pylint: disable=invalid-name
        vocab_size = 51864
        decoder_start_token_id = 50257

        for name in ort_names:
            if name in required_input_names:
                continue
            elif name == "vocab_mask":
                inputs.append(np.ones(vocab_size, dtype=np.int32))
            elif name == "prefix_vocab_mask":
                inputs.append(np.ones((batch_size, vocab_size), dtype=np.int32))
            elif name == "attention_mask":
                inputs.append(np.zeros((batch_size, N_MELS, N_FRAMES), dtype=np.int32))
            elif name == "decoder_input_ids":
                inputs.append(np.array([[decoder_start_token_id]], dtype=np.int32))
            elif name == "logits_processor":
                inputs.append(np.array([1], dtype=np.int32))
            else:
                raise NotImplementedError(f"'{name}' input is not supported")
        return inputs

    def test_sanity_check(self):
        """
        Test for sanity check
        """
        text_from_torch = (
            "<|0.00|> Продукт-менеджер, у него огромный опыт работы с GNI,<|4.00|>"
            "<|4.00|> с разными продуктами, с бизнесом, понимает ограничения,<|6.00|>"
            "<|6.00|> в том числе запросы, поэтому он здесь очень полезен.<|10.00|>"
            "<|10.00|> Наверное, там, если сделаешь небольшое интро про то,<|14.00|>"
            "<|14.00|> где работал, тоже, чтобы ребятам было комфортно,<|16.00|>"
            "<|16.00|> и потом расскажу о ребятах.<|18.00|>"
            "<|18.00|> Да-да-да. Последний год я работаю над консалтингом в AI.<|22.00|>"
            "<|22.00|> Вот мы с Асбером, с Яндексом, вот с Netology и с другими<|26.00|>"
            "<|26.00|> прикольными компаниями как раз повнедряли<|28.00|>"
            "<|28.00|>"
        )

        model_inputs = self._get_model_inputs(self.e2e_model_ort_session, self.raw_audio)
        text_from_onnx = self.e2e_model(*model_inputs)[0][0].strip()
        print(text_from_onnx, sep='\n')
        similarity = difflib.SequenceMatcher(None, text_from_torch, text_from_onnx).ratio()
        self.assertGreater(similarity, 0.95, f"Similarity ({similarity:.2f}) is lower than expected")

if __name__ == "__main__":
    unittest.main()
