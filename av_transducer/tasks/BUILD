cc_library(
    name = "ocr_graph",
    srcs = ["ocr_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//av_transducer/calculators:ocr_calculator",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        "@mediapipe//mediapipe/calculators/image:scale_image_calculator",
        "@mediapipe//mediapipe/calculators/util:from_image_calculator",
        "@mediapipe//mediapipe/framework:subgraph",
        "@mediapipe//mediapipe/framework/api2:builder",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/formats:image_frame",
    ],
    alwayslink = 1,
)

cc_library(
    name = "visual_graph",
    srcs = ["visual_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":ocr_graph",
        "//av_transducer/calculators:detection_calculator",
        "//av_transducer/calculators:lift_to_yuvimage_calculator",
        "//av_transducer/calculators:speaker_name_rect_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@mediapipe//mediapipe/calculators/image:yuv_to_image_calculator",
        "@mediapipe//mediapipe/framework:subgraph",
        "@mediapipe//mediapipe/framework/api2:builder",
        "@mediapipe//mediapipe/framework/formats:image",
    ],
    alwayslink = 1,
)

cc_library(
    name = "audio_graph",
    srcs = ["audio_graph.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//av_transducer/calculators:asr_calculator",
        "//av_transducer/calculators:audio_converter_calculator",
        "@mediapipe//mediapipe/framework:subgraph",
        "@mediapipe//mediapipe/framework/api2:builder",
    ],
    alwayslink = 1,
)
