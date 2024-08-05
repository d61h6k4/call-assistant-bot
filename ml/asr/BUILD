cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = ["model.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@vosk_api//:vosk",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    data = [
        "//ml/asr/models:vosk_models",
        "//testdata:test_audio",
    ],
    deps = [
        ":model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "model_benchmark",
    srcs = ["model_benchmark.cc"],
    data = [
        "//ml/asr/models:vosk_models",
        "//testdata:test_audio",
    ],
    tags = ["exclusive"],
    deps = [
        ":model",
        "@google_benchmark//:benchmark_main",
    ],
)
