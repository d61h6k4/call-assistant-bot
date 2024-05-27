load("@aspect_rules_lint//format:defs.bzl", "format_multirun")

# //:BUILD
filegroup(
    name = "clang_tidy_config",
    srcs = [".clang-tidy"],
    visibility = ["//visibility:public"],
)

format_multirun(
    name = "format",
    starlark = "@buildifier_prebuilt//:buildifier",
)
