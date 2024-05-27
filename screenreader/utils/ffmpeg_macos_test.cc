
#include "screenreader/utils/ffmpeg.h"
#include "gtest/gtest.h"
#include "absl/log/absl_log.h"
TEST(TestFFmpegUtils, CheckCaptureAudio) {
    auto res = aikit::utils::CaptureDevice("avfoundation", ":1");
    EXPECT_TRUE(res.ok()) << res.status().message();
}

TEST(TestFFmpegUtils, CheckCaptureScreen) {
    auto res = aikit::utils::CaptureDevice("avfoundation", "2:");
    EXPECT_TRUE(res.ok()) << res.status().message();
}
