load("@rules_proto//proto:defs.bzl", "proto_library")
load("@grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

proto_library(
    name = "evaluator_proto",
    srcs = ["evaluator.proto"],
    visibility = ["//meeting_bot:__subpackages__"],
)

py_proto_library(
    name = "evaluator_py_pb2",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":evaluator_proto"],
)

py_grpc_library(
    name = "evaluator_py_pb2_grpc",
    srcs = [":evaluator_proto"],
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":evaluator_py_pb2"],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [
        ":evaluator_py_pb2_grpc",
        "@grpc//src/python/grpcio/grpc:grpcio",
        "@pip//grpcio_health_checking",
        "@pip//picologging",
    ],
)
