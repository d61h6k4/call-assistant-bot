load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "root",
    srcs = ["root.py"],
    deps = [
        "@pip//fastapi",
        "@pip//fastui",
    ],
)

py_library(
    name = "landing_page",
    srcs = ["landing_page.py"],
    deps = [
        "@pip//fastapi",
        "@pip//fastui",
    ],
)

py_library(
    name = "auth",
    srcs = ["auth.py"],
    deps = [
        "@pip//fastapi",
        "@pip//fastui",
        "@pip//pyjwt",
    ],
)

py_library(
    name = "scheduler",
    srcs = ["scheduler.py"],
    deps = [
        ":auth",
        "@pip//fastapi",
        "@pip//fastui",
        "@pip//google_cloud_run",
    ],
)

py_library(
    name = "logging",
    srcs = ["logging.py"],
    deps = [
        "@pip//picologging",
        "@pip//python_json_logger",
    ],
)

py_binary(
    name = "cc_server",
    srcs = ["cc_server.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":auth",
        ":landing_page",
        ":logging",
        ":root",
        ":scheduler",
        "@pip//fastapi",
        "@pip//fastui",
        "@pip//picologging",
        "@pip//pydantic",
    ],
)

py_test(
    name = "scheduler_test",
    srcs = ["scheduler_test.py"],
    tags = ["manual"],
    deps = [":scheduler"],
)
