load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")


py_library(
    name = "logging",
    srcs = ["logging.py"],
    deps = [
        "@pip//picologging",
        "@pip//python_json_logger",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "logging_test",
    srcs = ["logging_test.py"],
    deps = [":logging"],
)
