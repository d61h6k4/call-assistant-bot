
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "gtest/gtest.h"

namespace aikit {
namespace {

TEST(FFMpegCaptureAudioCalculatorTest, AudioCheck) {
  auto calculator_node =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
          R"pb(
            calculator: "FFMPEGCaptureAudioCalculator"
            output_stream: "AUDIO:audio"
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());

  const auto &outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 1);
  const auto &audio = outputs.Tag("AUDIO").packets[0].Get<std::vector<float>>();

  EXPECT_EQ(audio.size(), 342);
}
} // namespace
} // namespace aikit
