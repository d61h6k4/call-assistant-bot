load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "asr_proto",
    srcs = ["asr.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "asr_cc_proto",
    deps = [":asr_proto"],
    visibility = ["//visibility:public"],
)