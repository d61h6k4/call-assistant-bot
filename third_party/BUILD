package(
    default_visibility = ["//visibility:public"],
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
