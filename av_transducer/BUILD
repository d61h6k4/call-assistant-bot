package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "av_transducer",
    srcs = ["main.cc"],
    data = [
        "//ml/detection/models:cdetr",
        "//ml/ocr/models:model",
    ],
    deps = [
        "//av_transducer/calculators:audio_converter_calculator",
        "//av_transducer/calculators:evaluator_client_calculator",
        "//av_transducer/calculators:ffmpeg_capture_audio_calculator",
        "//av_transducer/calculators:ffmpeg_capture_screen_calculator",
        "//av_transducer/calculators:ffmpeg_sink_video_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "//av_transducer/tasks:visual_graph",
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
        "//ml/ocr/models:model",
    ],
    deps = [
        "//av_transducer/calculators:ffmpeg_sink_video_calculator",
        "//av_transducer/calculators:ffmpeg_source_video_calculator",
        "//av_transducer/calculators:image_frame_to_video_frame_calculator",
        "//av_transducer/calculators:speaker_name_to_render_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "//av_transducer/tasks:visual_graph",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@mediapipe//mediapipe/calculators/core:packet_cloner_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:builder",
        "@mediapipe//mediapipe/util:color_cc_proto",
    ],
)
