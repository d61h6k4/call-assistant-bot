
sh_binary(
    name = "run_ffmpeg",
    srcs = ["run_ffmpeg.sh"],
    data = ["@ffmpeg//:ffmpeg_bin"],
    deps = ["@ffmpeg//:libffmpeg"],
    env = {
        "FFMPEG": "$(rootpath @ffmpeg//:ffmpeg_bin)"
    }
)
