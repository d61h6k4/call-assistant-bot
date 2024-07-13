from pprint import pprint
import argparse
import json
from functools import partial
from pathlib import Path
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection, Trainer
from transformers.image_transforms import center_to_corners_format
from datasets import Dataset
import torch
from transformers import TrainingArguments
from PIL import Image

import albumentations as A

import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import (
    _TORCHVISION_GREATER_EQUAL_0_8,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exported_json",
        help="Specify path to the json file exported from Label Studio.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_model",
        help="Specify path to the folder to store the trained model",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def show_example(example):
    from PIL import ImageDraw

    image = example["image"]
    annotations = example["objects"]
    draw = ImageDraw.Draw(image)

    label2id = {"speaker": 0, "participant": 1, "shared screen": 2}
    id2label = {v: k for k, v in label2id.items()}
    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        class_idx = annotations["category"][i]
        x, y, w, h = tuple(box)

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), id2label[class_idx], fill="white")

    image.show()


def gen_dataset(exported_json):
    tasks = json.loads(exported_json.read_text())
    for task in tasks:
        image_id = task["id"]
        image_path = "/" + task["data"]["image"].split("=")[1]
        image = Image.open(image_path)
        width = image.size[0]
        height = image.size[1]

        label_to_category = {"speaker": 0, "participant": 1, "shared screen": 2}
        objects = {"id": [], "bbox": [], "category": [], "area": []}
        # TODO(d61h6k4) Choose ground truth or last updated
        annotation = task["annotations"][0]
        for result in annotation["result"]:
            objects["id"].append(result["id"])
            objects["bbox"].append(
                [
                    result["value"]["x"] / 100.0 * result["original_width"],
                    result["value"]["y"] / 100.0 * result["original_height"],
                    result["value"]["width"] / 100.0 * result["original_width"],
                    result["value"]["height"] / 100.0 * result["original_height"],
                ]
            )
            objects["area"].append(result["value"]["width"] * result["value"]["height"])
            objects["category"].append(
                label_to_category[result["value"]["rectanglelabels"][0]]
            )
        example = {
            "image_id": image_id,
            "image": image,
            "width": width,
            "height": height,
            "objects": objects,
        }
        yield example


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples, transform, image_processor, return_pixel_mask=False
):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(
        images=images, annotations=annotations, return_tensors="pt"
    )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
        )
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(
        classes, map_per_class, mar_100_per_class
    ):
        class_name = (
            id2label[class_id.item()] if id2label is not None else class_id.item()
        )
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()
    ds = Dataset.from_generator(
        gen_dataset, gen_kwargs={"exported_json": args.exported_json}
    ).train_test_split(test_size=0.2)

    model_name = "microsoft/conditional-detr-resnet-50"
    image_processor = AutoImageProcessor.from_pretrained(
        model_name, do_resize=True, size={"shortest_edge": 504, "longest_edge": 896}
    )

    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=25
        ),
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=image_processor,
    )
    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=image_processor,
    )
    train_ds = ds["train"].with_transform(train_augment_and_transform)
    val_ds = ds["test"].with_transform(validation_transform_batch)

    label2id = {"speaker": 0, "participant": 1, "shared screen": 2}
    id2label = {v: k for k, v in label2id.items()}

    eval_compute_metrics_fn = partial(
        compute_metrics,
        image_processor=image_processor,
        id2label=id2label,
        threshold=0.0,
    )

    model = ConditionalDetrForObjectDetection.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="detr_finetuned_cppe5",
        num_train_epochs=24,
        fp16=False,
        per_device_train_batch_size=12,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()
    trainer.save_model(args.output_model)

    metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="test")
    pprint(metrics)


if __name__ == "__main__":
    main()
