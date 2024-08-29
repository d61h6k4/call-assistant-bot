load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")


py_binary(
    name = "converter",
    srcs = ["converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
        "@pip//onnxscript",
        "@pip//optimum",
        "@pip//torch",
        "@pip//librosa",
        "@pip//soundfile",
        "@pip//psutil",
        "@pip//protobuf",
        "//third_party:libffmpeg",
        "@pip//transformers",
    ],
)


py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    data = [
        "//testdata:test_audio",
        "//ml/whisper/models:model",
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
        "@pip//torch",
        "@pip//transformers",
        "@pip//librosa",
        "@pip//soundfile",
        "//third_party:libffmpeg",
    ],
)