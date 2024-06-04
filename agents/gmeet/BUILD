load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "gmeet",
    srcs = ["gmeet.py"],
    deps = [
        "@pip//nodriver",
        "@pip//picologging",
    ],
)
