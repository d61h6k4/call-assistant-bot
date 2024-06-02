package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "screenreader",
    srcs = ["main.cc"],
    deps = [
        "//screenreader/calculators:ffmpeg_capture_audio_calculator",
        "//screenreader/calculators:ffmpeg_sink_video_calculator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:builder",
    ],
)
