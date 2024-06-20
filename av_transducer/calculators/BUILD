package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ffmpeg_capture_audio_calculator",
    srcs = ["ffmpeg_capture_audio_calculator.cc"],
    deps = [
        "//av_transducer/utils:audio",
        "//av_transducer/utils:container",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "ffmpeg_capture_screen_calculator",
    srcs = ["ffmpeg_capture_screen_calculator.cc"],
    deps = [
        "//av_transducer/utils:container",
        "//av_transducer/utils:video",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "ffmpeg_sink_video_calculator",
    srcs = ["ffmpeg_sink_video_calculator.cc"],
    deps = [
        "//av_transducer/utils:audio",
        "//av_transducer/utils:container",
        "//av_transducer/utils:video",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:yuv_image",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",

"@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
    ],
    alwayslink = True,
)

cc_library(
    name = "audio_converter_calculator",
    srcs = ["audio_converter_calculator.cc"],
    deps = [
        "//av_transducer/utils:audio",
        "//av_transducer/utils:converter",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)


cc_library(
    name = "video_converter_calculator",
    srcs = ["video_converter_calculator.cc"],
    deps = [
        "//av_transducer/utils:video",
        "//av_transducer/utils:converter",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_test(
    name = "ffmpeg_capture_audio_calculator_test",
    srcs = ["ffmpeg_capture_audio_calculator_test.cc"],
    tags = ["manual"],
    deps = [
        ":ffmpeg_capture_audio_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "ffmpeg_capture_screen_calculator_test",
    srcs = ["ffmpeg_capture_screen_calculator_test.cc"],
    tags = ["manual"],
    deps = [
        ":ffmpeg_capture_screen_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "ffmpeg_sink_video_calculator_test",
    srcs = ["ffmpeg_sink_video_calculator_test.cc"],
    deps = [
        ":ffmpeg_sink_video_calculator",
        "//third_party:libffmpeg",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "audio_converter_calculator_test",
    srcs = ["audio_converter_calculator_test.cc"],
    data = [
        "//testdata:test_videos",
    ],
    deps = [
        ":audio_converter_calculator",
        "//av_transducer/utils:container",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "video_converter_calculator_test",
    srcs = ["video_converter_calculator_test.cc"],
    data = [
        "//testdata:test_videos",
    ],
    deps = [
        ":video_converter_calculator",
        "//av_transducer/utils:container",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)
