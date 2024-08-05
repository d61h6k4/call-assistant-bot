load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")


py_binary(
    name = "converter",
    srcs = ["converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    data = ["//testdata:test_images"],
    deps = [
        "@pip//onnxtr",
        "@pip//numpy",
        "@pip//pillow",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    data = [
        "//testdata:test_images",
        "//ml/ocr/models:model",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":converter",
    ],
)
