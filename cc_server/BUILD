
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "landing_page",
    srcs = ["landing_page.py"],
    deps = [
        "@pip//fastapi",
        "@pip//fastui",
    ]
)

py_binary(
    name = "cc_server",
    srcs = ["cc_server.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":landing_page",
        "@pip//fastapi",
        "@pip//pydantic",
        "@pip//fastui",
        "@pip//picologging",
    ],
)
