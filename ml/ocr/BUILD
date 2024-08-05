load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = ["model.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ml/ocr/models:vocab",
        "//third_party:libonnxruntime",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//ml/ocr/models:model",
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
        "//ml/ocr/models:model",
        "//testdata:test_images",
    ],
    tags = ["exclusive"],
    deps = [
        ":model",
        "//third_party:opencv",
        "@google_benchmark//:benchmark_main",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    data = ["//testdata:test_images"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//easyocr",
        "@pip//numpy",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
        "@pip//pillow",
        "@pip//torch",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    data = [
        "//ml/ocr/models:model",
        "//testdata:test_images",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":converter",
    ],
)
