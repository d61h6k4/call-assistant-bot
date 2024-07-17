

cc_library(
    name = "visual_graph",
    srcs = ["visual_graph.cc"],
    deps = [
        "//av_transducer/calculators:cdetr_calculator",
        "//av_transducer/calculators:lift_to_yuvimage_calculator",
        "//av_transducer/calculators:video_converter_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@mediapipe//mediapipe/calculators/image:yuv_to_image_calculator",
        "@mediapipe//mediapipe/framework/api2:builder",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework:subgraph",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
