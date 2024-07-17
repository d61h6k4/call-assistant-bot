load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
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

py_library(
    name = "xvfbwrapper",
    srcs = ["xvfbwrapper.py"],
)

py_library(
    name = "google_cloud",
    srcs = [
        "google_cloud.py",
    ],
    deps = [
        "@pip//google_auth",
        "@pip//google_cloud_storage",
    ],
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
    visibility = ["//visibility:public"],
    deps = [
        ":google_cloud",
        ":meeting_bot_py_pb2",
        ":meeting_bot_py_pb2_grpc",
        ":xvfbwrapper",
        "//meeting_bot/articulator:articulator_py_pb2",
        "//meeting_bot/articulator:articulator_py_pb2_grpc",
        "//meeting_bot/evaluator:evaluator_py_pb2",
        "//meeting_bot/evaluator:evaluator_py_pb2_grpc",
        "//meeting_bot/perceiver:perceiver_py_pb2",
        "//meeting_bot/perceiver:perceiver_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
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
    tags = ["manual"],
    deps = [
        ":meeting_bot_py_pb2_grpc",
    ],
)

py_test(
    name = "google_cloud_test",
    srcs = ["google_cloud_test.py"],
    data = ["//testdata:test_videos"],
    tags = ["manual"],
    deps = [":google_cloud"],
)
