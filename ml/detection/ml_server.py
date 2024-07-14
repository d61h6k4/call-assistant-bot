from typing import NamedTuple
from transformers import AutoImageProcessor
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrObjectDetectionOutput,
)
from onnxruntime_extensions import OrtPyFunction
import httpx
import uvicorn
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

_LOGGER = logging.getLogger("uvicorn.error")
LB_SERVER = "http://0.0.0.0:8080"
ACCESS_TOKEN = None


class Model:
    def __init__(self):
        self.onnx_model = OrtPyFunction.from_model("ml/detection/models/model.onnx")

    def __call__(self, image_name: str, image: Image.Image):
        onnx_output = self.onnx_model(np.transpose(np.array(image), (2, 0, 1)))

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
                    "x": float((onnx_output[idx][0] - 0.5 * onnx_output[idx][2]) * 100),
                    "y": float((onnx_output[idx][1] - 0.5 * onnx_output[idx][3]) * 100),
                    "width": float(onnx_output[idx][2] * 100),
                    "height": float(onnx_output[idx][3] * 100),
                    "rectanglelabels": [id2label[onnx_output[idx][5]]],
                    "score": float(onnx_output[idx][4]),
                },
            }
            for idx in range(len(onnx_output))
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
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    predictions = []
    for task in body.get("tasks", []):
        path_to_image = task["data"]["image"]
        r = httpx.get(
            f"{LB_SERVER}{path_to_image}",
            headers={"Authorization": f"Token {ACCESS_TOKEN}"},
        )
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
