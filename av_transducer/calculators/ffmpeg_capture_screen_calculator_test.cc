#include <memory>

#include "av_transducer/utils/video.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "gtest/gtest.h"

namespace aikit {
namespace {

TEST(FFMPEGCaptureScreenCalculatorTest, HeaderCheck) {
  auto calculator_node =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "FFMPEGCaptureScreenCalculator"
            output_stream: "VIDEO:video_frames"
            output_side_packet: "VIDEO_HEADER:video_header"
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());
  auto &video_header = runner.OutputSidePackets()
                           .Tag("VIDEO_HEADER")
                           .Get<media::VideoStreamParameters>();

  EXPECT_EQ(video_header.width, 3840);
}

TEST(FFMpegCaptureScreenCalculatorTest, VideoCheck) {
  auto calculator_node =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "FFMPEGCaptureScreenCalculator"
            output_side_packet: "VIDEO_HEADER:video_header"
            output_stream: "VIDEO:video"
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());

  auto &outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto &video=
      outputs.Tag("VIDEO").packets[0].Get<aikit::media::VideoFrame>();
  EXPECT_GE(video.GetPTS(), 0);
}
} // namespace
} // namespace aikit
