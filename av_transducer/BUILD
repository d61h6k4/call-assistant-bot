package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "av_transducer",
    srcs = ["main.cc"],
    data = [
        "//ml/detection/models:cdetr",
    ],
    deps = [
        "//av_transducer/tasks:visual_graph",
        "//av_transducer/calculators:evaluator_client_calculator",
        "//av_transducer/calculators:audio_converter_calculator",
        "//av_transducer/calculators:ffmpeg_capture_audio_calculator",
        "//av_transducer/calculators:ffmpeg_capture_screen_calculator",
        "//av_transducer/calculators:ffmpeg_sink_video_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:builder",
    ],
)

cc_binary(
    name = "debug",
    srcs = ["debug.cc"],
    data = [
        "//ml/detection/models:cdetr",
    ],
    deps = [
        "//av_transducer/tasks:visual_graph",
        "//av_transducer/calculators:dumper_calculator",
        "//av_transducer/calculators:ffmpeg_source_video_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:builder",
    ],
)
