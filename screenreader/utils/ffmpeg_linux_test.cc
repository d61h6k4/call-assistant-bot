
#include "screenreader/utils/ffmpeg.h"
#include "gtest/gtest.h"

TEST(TestFFmpegUtils, CheckCaptureAudio) {
  auto res = aikit::utils::CaptureDevice("pulse", "default");
  EXPECT_TRUE(res.ok()) << res.status().message();
}

TEST(TestFFmpegUtils, CheckCaptureScreen) {
  auto res = aikit::utils::CaptureDevice("x11grab", ":0.0+100,200");
  EXPECT_TRUE(res.ok()) << res.status().message();
}
