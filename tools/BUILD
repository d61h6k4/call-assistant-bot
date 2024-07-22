load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "sync",
    srcs = ["sync.py"],
    data = [
        "//ml/detection/models:cdetr",
        "//testdata:test_images",
        "//testdata:test_videos",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//google_auth",
        "@pip//google_cloud_storage",
        "@pip//picologging",
    ],
)
