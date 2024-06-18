
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "server",
    srcs = ["server.py"],
    data = ["//data:client_secret_json"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@pip//google_auth",
        "@pip//google_auth_oauthlib",
        "@pip//google_apps_meet",
        "@pip//google_cloud_pubsub",
        "@pip//picologging",
    ],
)
