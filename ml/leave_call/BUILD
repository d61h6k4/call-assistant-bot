load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//picologging",
        "@pip//river",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "//testdata:evaluator_logs",
    ],
    deps = [
        ":model",
        "//meeting_bot/evaluator:evaluator_py_pb2",
    ],
)
