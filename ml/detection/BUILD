load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = ["model.h"],
    visibility = ["//visibility:public"],
    deps = ["//third_party:libonnxruntime"],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//ml/detection/models:cdetr",
        "//testdata:test_images",
    ],
    deps = [
        ":model",
        "//third_party:opencv",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_benchmark",
    srcs = ["model_benchmark.cc"],
    data = [
        "//ml/detection/models:cdetr",
        "//testdata:test_images",
    ],
    deps = [
        ":model",
        "//third_party:opencv",
        "@google_benchmark//:benchmark_main",
    ],
)

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
    tags = ["manual"],
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
    data = [
        "//testdata:test_images",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":ml_server",
    ],
)
