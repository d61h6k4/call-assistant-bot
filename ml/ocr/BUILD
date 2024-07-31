load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")


py_binary(
    name = "converter",
    srcs = ["converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_binary(
    name = "preprocessing_converter",
    srcs = ["preprocessing_converter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//numpy",
        "@pip//onnx",
        "@pip//onnxruntime",
        "@pip//onnxruntime_extensions",
    ],
)

cc_library(
    name = "processor",
    srcs = ["processor.cc"],
    hdrs = ["processor.h"],
    deps = [
        "@onnxruntime_genai//:genai",
    ],
)

cc_library(
    name = "encoder_kvcache",
    srcs = ["encoder_kvcache.cc"],
    hdrs = ["encoder_kvcache.h"],
    deps = [
        "@onnxruntime_genai//:genai",
    ],
)

cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    data = [
        "//models/florence2/data:decoder",
        "//models/florence2/data:encoder",
        "//models/florence2/data:genai_config",
        "//models/florence2/data:image_processor",
        "//models/florence2/data:tokenizer",
        "//models/florence2/data:tokenizer_config",
        "//models/florence2/data:vision_encoder",
    ],
    deps = [
        ":encoder_kvcache",
        "@onnxruntime_genai//:genai",
    ],
)

cc_test(
    name = "processor_test",
    srcs = ["processor_test.cc"],
    data = ["//models/florence2/data:test_image"],
    deps = [
        ":model",
        ":processor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = ["//models/florence2/data:test_image"],
    deps = [
        ":model",
        ":processor",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tokenizer_test",
    srcs = ["tokenizer_test.cc"],
    data = [
        "//models/florence2/data:tokenizer_config",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@onnxruntime_extensions//:shared",
    ],
)

cc_test(
    name = "image_processor_test",
    srcs = ["image_processor_test.cc"],
    data = [
        "//models/florence2/data:image_processor",
        "//models/florence2/data:test_image",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@onnxruntime_extensions//:shared",
    ],
)


cc_binary(
    name = "example",
    srcs = ["example.cc"],
    data = [
        "//models/florence2/data:test_image",
    ],
    deps = [
        ":model",
        ":processor",
    ],
)
