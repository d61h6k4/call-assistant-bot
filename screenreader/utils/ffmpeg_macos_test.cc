
#include "screenreader/utils/ffmpeg.h"
#include "gtest/gtest.h"

TEST(TestFFmpegUtils, CheckCaptureAudio) {
    auto res = aikit::utils::CaptureDevice("avfoundation", ":1");
    EXPECT_TRUE(res.ok()) << res.message();
}

TEST(TestFFmpegUtils, CheckCaptureScreen) {
    auto res = aikit::utils::CaptureDevice("avfoundation", "2:");
    EXPECT_TRUE(res.ok()) << res.message();
}
