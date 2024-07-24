load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")
load("@com_github_grpc_grpc//bazel:grpc_build_system.bzl", "grpc_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "evaluator_proto",
    srcs = ["evaluator.proto"],
    visibility = ["//meeting_bot:__subpackages__"],
)

py_proto_library(
    name = "evaluator_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":evaluator_proto"],
)

py_grpc_library(
    name = "evaluator_py_pb2_grpc",
    srcs = [":evaluator_proto"],
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":evaluator_py_pb2"],
)

grpc_proto_library(
    name = "evaluator_proto_grpc",
    srcs = ["evaluator.proto"],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [
        ":evaluator_py_pb2_grpc",
        "//meeting_bot/common:logging",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@pip//grpcio_health_checking",
        "@pip//picologging",
    ],
)
