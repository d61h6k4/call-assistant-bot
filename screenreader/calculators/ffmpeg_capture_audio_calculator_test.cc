
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "screenreader/utils/audio.h"
#include "gtest/gtest.h"

namespace aikit {
namespace {

TEST(FFMpegCaptureAudioCalculatorTest, AudioHeaderCheck) {
  auto calculator_node =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
                calculator: "FFMPEGCaptureAudioCalculator"
                output_side_packet: "AUDIO_HEADER:audio_header"
                output_stream: "AUDIO:audio"
              )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());
  auto &audio_header = runner.OutputSidePackets()
                           .Tag("AUDIO_HEADER")
                           .Get<media::AudioStreamParameters>();
  EXPECT_GT(audio_header.frame_size, 0);
}

TEST(FFMpegCaptureAudioCalculatorTest, AudioCheck) {
  auto calculator_node =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "FFMPEGCaptureAudioCalculator"
            output_side_packet: "AUDIO_HEADER:audio_header"
            output_stream: "AUDIO:audio"
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());

  const auto &outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto &audio = outputs.Tag("AUDIO").packets[0].Get<std::vector<float>>();

  EXPECT_GT(audio.size(), 0);
}
} // namespace
} // namespace aikit
