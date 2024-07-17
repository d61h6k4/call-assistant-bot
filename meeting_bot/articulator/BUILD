load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

proto_library(
    name = "articulator_proto",
    srcs = ["articulator.proto"],
    visibility = ["//meeting_bot:__subpackages__"],
)

py_proto_library(
    name = "articulator_py_pb2",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":articulator_proto"],
)

py_grpc_library(
    name = "articulator_py_pb2_grpc",
    srcs = [":articulator_proto"],
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":articulator_py_pb2"],
)

py_library(
    name = "gmeet",
    srcs = ["gmeet.py"],
    visibility = ["//meeting_bot/articulator:__subpackages__"],
    deps = [
        "@pip//nodriver",
        "@pip//picologging",
    ],
)

py_binary(
    name = "articulator",
    srcs = ["articulator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [
        ":articulator_py_pb2_grpc",
        ":gmeet",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@pip//grpcio_health_checking",
        "@pip//nodriver",
        "@pip//picologging",
    ],
)
