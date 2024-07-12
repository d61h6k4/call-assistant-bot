load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")



py_binary(
    name = "ml_server",
    srcs = ["ml_server.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    data = [
        "//ml/detection/models:cdetr",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//fastapi",
        "@pip//uvicorn",
        "@pip//pydantic",
        "@pip//pillow",
        "@pip//httpx",
        "@pip//numpy",
        "@pip//torch",
        "@pip//transformers",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
    ],
)


py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//transformers",
        "@pip//torch",
        "@pip//pillow",
        "@pip//timm",
        "@pip//accelerate",
        "@pip//datasets",
        "@pip//albumentations",
        "@pip//torchmetrics",
        "@pip//pycocotools",
        "@pip//matplotlib",
        "@pip//torchvision",
    ],
)


py_binary(
    name = "converter",
    srcs = ["converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//transformers",
        "@pip//torch",
        "@pip//pillow",
        "@pip//timm",
        "@pip//accelerate",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxscript",
        "@pip//optimum",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//transformers",
        "@pip//torch",
        "@pip//pillow",
        "@pip//timm",
        "@pip//accelerate",
        "@pip//onnx",
        "@pip//onnxscript",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
    ],
)


py_test(
    name = "ml_server_test",
    srcs = ["ml_server_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":ml_server",
    ],
)
