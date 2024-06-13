load("@rules_proto//proto:defs.bzl", "proto_library")
load("@grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

proto_library(
    name = "meeting_bot_proto",
    srcs = ["meeting_bot.proto"],
)

py_proto_library(
    name = "meeting_bot_py_pb2",
    deps = [":meeting_bot_proto"],
)

py_grpc_library(
    name = "meeting_bot_py_pb2_grpc",
    srcs = [":meeting_bot_proto"],
    deps = [":meeting_bot_py_pb2"],
)

py_binary(
    name = "meeting_bot",
    srcs = ["meeting_bot.py"],
    data = [
        "//meeting_bot/articulator",
        "//meeting_bot/evaluator",
        "//meeting_bot/perceiver",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":meeting_bot_py_pb2",
        ":meeting_bot_py_pb2_grpc",
        "//meeting_bot/articulator:articulator_py_pb2",
        "//meeting_bot/articulator:articulator_py_pb2_grpc",
        "//meeting_bot/evaluator:evaluator_py_pb2",
        "//meeting_bot/evaluator:evaluator_py_pb2_grpc",
        "//meeting_bot/perceiver:perceiver_py_pb2",
        "//meeting_bot/perceiver:perceiver_py_pb2_grpc",
        "@grpc//src/python/grpcio/grpc:grpcio",
        "@pip//grpcio_health_checking",
        "@pip//picologging",
        "@rules_python//python/runfiles",
    ],
)

py_test(
    name = "meeting_bot_test",
    size = "small",
    srcs = ["meeting_bot_test.py"],
    data = [
        ":meeting_bot",
    ],
    deps = [
        ":meeting_bot_py_pb2_grpc",
    ],
)