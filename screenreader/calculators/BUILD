package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ffmpeg_capture_audio_calculator",
    srcs = ["ffmpeg_capture_audio_calculator.cc"],
    deps = [
        "//screenreader/utils:audio",
        "//screenreader/utils:container",
        "//screenreader/utils:converter",
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
        "//screenreader/utils:ffmpeg",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:image_format_cc_proto",
        "@mediapipe//mediapipe/framework/formats:video_stream_header",
        "@mediapipe//mediapipe/framework/formats:yuv_image",
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
        "//screenreader/utils:audio",
        "//screenreader/utils:container",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:yuv_image",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_test(
    name = "ffmpeg_capture_audio_calculator_test",
    srcs = ["ffmpeg_capture_audio_calculator_test.cc"],
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
    deps = [
        ":ffmpeg_capture_screen_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/calculators/image:yuv_to_image_calculator",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:opencv_imgcodecs",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
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
        "@mediapipe//mediapipe/calculators/image:yuv_to_image_calculator",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/formats:yuv_image",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)
