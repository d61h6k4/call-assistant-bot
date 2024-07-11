load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@pip//:requirements.bzl", "requirement")


py_binary(
    name = "label_studio",
    srcs = ["label_studio.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//setuptools",
        "@pip//label_studio",
    ],
)
