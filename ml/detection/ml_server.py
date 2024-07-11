from typing import NamedTuple
from transformers import AutoImageProcessor
from transformers.models.conditional_detr.modeling_conditional_detr import ConditionalDetrObjectDetectionOutput
from onnxruntime_extensions import OrtPyFunction
import httpx
import uvicorn
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch

_LOGGER = logging.getLogger("uvicorn.error")
LB_SERVER = "http://0.0.0.0:8080"
ACCESS_TOKEN = None

class Model:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("ml/detection/models/preprocessor_config.json")
        self.onnx_model = OrtPyFunction.from_model("ml/detection/models/model.onnx")

    def __call__(self, image_name: str, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="np")
        onnx_output = self.onnx_model(inputs["pixel_values"])

        target_sizes = torch.tensor([image.size[::-1]])
        output = ConditionalDetrObjectDetectionOutput(logits=torch.tensor(onnx_output[0]), pred_boxes=torch.tensor(onnx_output[1]))
        results = self.processor.post_process_object_detection(output, target_sizes=target_sizes, threshold=0.29)[0]

        _LOGGER.debug(results)

        label2id = {"speaker": 0, "participant": 1, "shared screen": 2}
        id2label = {v: k for k, v in label2id.items()}
        width = image.size[0]
        height = image.size[1]
        return [
            {
                "id": f"{image_name}-{idx}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": results["boxes"][idx][0].item() / width * 100,
                    "y": results["boxes"][idx][1].item() / height * 100,
                    "width": (results["boxes"][idx][2].item() - results["boxes"][idx][0].item()) / width * 100,
                    "height": (results["boxes"][idx][3].item() - results["boxes"][idx][1].item()) / height * 100,
                    "rectanglelabels": [id2label[results["labels"][idx].item()]],
                    "score": results["scores"][idx].item()
                },
            }
            for idx in range(len(results["scores"]))
        ]

class HealthResponse(BaseModel):
    status: str
    model_class: str

    @classmethod
    def ok(cls):
        return cls(status="UP", model_class="ConditionalDETR")


app = FastAPI()
MODEL = Model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.post("/predict")
async def predict(request: Request):
    body =  await request.json()

    predictions = []
    for task in body.get("tasks", []):
        path_to_image = task["data"]["image"]
        r = httpx.get(f"{LB_SERVER}{path_to_image}", headers={"Authorization": f"Token {ACCESS_TOKEN}"})
        r.raise_for_status()
        image = Image.open(r)
        result = MODEL(path_to_image.split("/")[-1].split("-")[0], image)

        predictions.append({"model_version": "CDETR", "score": 1.0, "result": result})

    _LOGGER.info(predictions)
    return {"results": predictions}

@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse.ok()

@app.post("/setup")
async def setup(request: Request):
    body = await request.json()
    global ACCESS_TOKEN
    ACCESS_TOKEN = body["access_token"]
    _LOGGER.debug(body)
    return {"model_version": "CDETR"}

if __name__ == "__main__":
    uvicorn.run("ml.detection.ml_server:app", port=5000, log_level="debug")
