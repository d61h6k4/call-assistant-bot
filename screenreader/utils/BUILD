cc_library(
    name = "ffmpeg",
    srcs = ["ffmpeg.cc"],
    hdrs = ["ffmpeg.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:libffmpeg",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "ffmpeg_test",
    srcs = select({
        "@platforms//os:macos": [
            "ffmpeg_macos_test.cc",
        ],
        "@platforms//os:linux": [
            "ffmpeg_linux_test.cc",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":ffmpeg",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
