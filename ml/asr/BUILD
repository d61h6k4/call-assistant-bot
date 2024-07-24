cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = ["model.h"],
    visibility = ["//visibility:public"],
    alwayslink = 0,
    deps = [
        "@vosk_api//:vosk",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
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

cc_binary(
    name = "example",
    srcs = ["example.cc"],
    data = [
        "//ml/asr/models:vosk_models",
        "//testdata:test_audio",
    ],
    deps = [
        "@vosk_api//:vosk",
        "@nlohmann_json//:json",
    ],
)