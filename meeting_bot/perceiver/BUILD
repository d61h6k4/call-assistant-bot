load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "av_transducer_operator",
    srcs = ["av_transducer_operator.py"],
    data = [
        "//av_transducer",
    ],
    visibility = ["//meeting_bot:__subpackages__"],
)

py_test(
    name = "av_transducer_operator_test",
    srcs = ["av_transducer_operator_test.py"],
    deps = [":av_transducer_operator"],
)

proto_library(
    name = "perceiver_proto",
    srcs = ["perceiver.proto"],
    visibility = ["//meeting_bot:__subpackages__"],
)

py_proto_library(
    name = "perceiver_py_pb2",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":perceiver_proto"],
)

py_grpc_library(
    name = "perceiver_py_pb2_grpc",
    srcs = [":perceiver_proto"],
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [":perceiver_py_pb2"],
)

py_binary(
    name = "perceiver",
    srcs = ["perceiver.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//meeting_bot:__subpackages__"],
    deps = [
        ":av_transducer_operator",
        ":perceiver_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@pip//grpcio_health_checking",
        "@pip//picologging",
    ],
)
