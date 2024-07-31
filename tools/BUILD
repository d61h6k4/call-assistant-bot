load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "sync",
    srcs = ["sync.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//google_auth",
        "@pip//google_cloud_storage",
        "@pip//picologging",
    ],
)
