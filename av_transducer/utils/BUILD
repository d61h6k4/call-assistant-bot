cc_library(
    name = "audio",
    srcs = ["audio.cc"],
    hdrs = ["audio.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:libffmpeg",
        "//third_party:libyuv",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "audio_test",
    srcs = ["audio_test.cc"],
    deps = [
        ":audio",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "container",
    srcs = ["container.cc"],
    hdrs = ["container.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":audio",
        "//third_party:libffmpeg",
        "//third_party:libyuv",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "container_test",
    srcs = ["container_test.cc"],
    data = ["//testdata:test_videos"],
    deps = [
        ":container",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":audio",
        "//third_party:libffmpeg",
        "//third_party:libyuv",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "converter_test",
    srcs = ["converter_test.cc"],
    data = ["//testdata:test_videos"],
    deps = [
        ":container",
        ":converter",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "capture_device_test",
    srcs = select({
        "@platforms//os:macos": [
            "capture_device_macos_test.cc",
        ],
        "@platforms//os:linux": [
            "capture_device_linux_test.cc",
        ],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = [
        ":container",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
