load("@aspect_rules_lint//format:defs.bzl", "format_multirun")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

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

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
        "//ml/...": "",
        "//av_transducer/...": "",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
    # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)
