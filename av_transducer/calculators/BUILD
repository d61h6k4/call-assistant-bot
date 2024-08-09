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
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
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
        "//av_transducer/utils:converter",
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
    name = "lift_to_yuvimage_calculator",
    srcs = ["lift_to_yuvimage_calculator.cc"],
    deps = [
        "//av_transducer/utils:video",
        "@//third_party:libyuv",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:yuv_image",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "detection_calculator",
    srcs = ["detection_calculator.cc"],
    deps = [
        "//ml/detection:model",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "ffmpeg_source_video_calculator",
    srcs = ["ffmpeg_source_video_calculator.cc"],
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
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "evaluator_client_calculator",
    srcs = ["evaluator_client_calculator.cc"],
    deps = [
        "//meeting_bot/evaluator:evaluator_proto_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "speaker_name_rect_calculator",
    srcs = ["speaker_name_rect_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/formats:rect_cc_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "ocr_calculator",
    srcs = ["ocr_calculator.cc"],
    deps = [
        "//ml/ocr:model",
        "//third_party:opencv",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "image_frame_to_video_frame_calculator",
    srcs = ["image_frame_to_video_frame_calculator.cc"],
    deps = [
        "//av_transducer/utils:video",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
    ],
    alwayslink = True,
)

cc_library(
    name = "speaker_name_to_render_calculator",
    srcs = ["speaker_name_to_render_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/api2:node",
        "@mediapipe//mediapipe/framework/api2:packet",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
        "@mediapipe//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = True,
)

cc_test(
    name = "image_frame_to_video_frame_calculator_test",
    srcs = ["image_frame_to_video_frame_calculator_test.cc"],
    data = [
        "//testdata:test_images",
    ],
    deps = [
        ":image_frame_to_video_frame_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "ocr_calculator_test",
    srcs = ["ocr_calculator_test.cc"],
    data = [
        "//ml/ocr/models:model",
        "//testdata:test_images",
    ],
    deps = [
        ":ocr_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "detection_calculator_test",
    srcs = ["detection_calculator_test.cc"],
    data = [
        "//ml/detection/models:cdetr",
        "//testdata:test_images",
    ],
    deps = [
        ":detection_calculator",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:gtest",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
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
