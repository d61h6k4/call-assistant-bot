load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "ml_server",
    srcs = ["ml_server.py"],
    data = [
        "//ml/detection/models:cdetr",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//fastapi",
        "@pip//httpx",
        "@pip//numpy",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
        "@pip//pillow",
        "@pip//pydantic",
        "@pip//torch",
        "@pip//transformers",
        "@pip//uvicorn",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//accelerate",
        "@pip//albumentations",
        "@pip//datasets",
        "@pip//matplotlib",
        "@pip//pillow",
        "@pip//pycocotools",
        "@pip//timm",
        "@pip//torch",
        "@pip//torchmetrics",
        "@pip//torchvision",
        "@pip//transformers",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//accelerate",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
        "@pip//onnxscript",
        "@pip//optimum",
        "@pip//pillow",
        "@pip//timm",
        "@pip//torch",
        "@pip//transformers",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    data = [
        "//testdata:test_images",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":converter",
        "@pip//accelerate",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
        "@pip//onnxscript",
        "@pip//pillow",
        "@pip//timm",
        "@pip//torch",
        "@pip//transformers",
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
