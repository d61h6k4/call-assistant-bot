package(
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "osx_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

alias(
    name = "libonnxruntime",
    actual = select({
        ":osx_arm64": "@onnxruntime_osx_arm64//:libonnxruntime",
        ":linux_aarch64": "@onnxruntime_linux_aarch64//:libonnxruntime",
        "//conditions:default": "@onnxruntime_linux_x64//:libonnxruntime",
    }),
)

cc_library(
    name = "libffmpeg",
    linkopts = select({
        "@platforms//os:macos": [
            "-framework AVFoundation",
            "-framework CoreMedia",
            "-framework AppKit",
            "-framework CoreFoundation",
            "-framework CoreGraphics",
            "-framework OpenGL",
            "-framework Security",
            "-framework CoreAudio",
            "-framework CoreImage",
            "-framework CoreVideo",
            "-framework AudioToolbox",
            "-framework VideoToolbox",
            "-framework Metal",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = ["@ffmpeg//:libffmpeg"],
)

alias(
    name = "libyuv",
    actual = "@libyuv//:libyuv",
)

cc_library(
    name = "opencv",
    visibility = ["//visibility:public"],
    deps = ["@opencv"],
)
