from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import numpy as np

from onnxruntime_extensions import OrtPyFunction


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", attn_implementation="sdpa", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

    prompt = "<CAPTION>"

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw).resize((768, 768), Image.Resampling.BICUBIC)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    print(model._encode_image(inputs["pixel_values"]))
    print(model.get_input_embeddings()(inputs["input_ids"]))
#
#    generated_ids = model.generate(
#        input_ids=inputs["input_ids"],
#        pixel_values=inputs["pixel_values"],
#        max_new_tokens=1024,
#        do_sample=False,
#        num_beams=3
#    )
#    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#
#    parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
#
#    print(parsed_answer)
# new_model_path = "/tmp/vision_encoder_with_postprocessing.onnx"
# m = OrtPyFunction.from_model(new_model_path)
# x = np.array(image).transpose([2, 0, 1])
# r = m(x)
# print(r)

# image.save("/tmp/car.jpg")
