#include <memory>

#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
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
            output_stream: "VIDEO_PRESTREAM:video_header"
          )pb");
  mediapipe::CalculatorRunner runner(calculator_node);

  MP_ASSERT_OK(runner.Run());

  const auto &outputs = runner.Outputs();
  EXPECT_EQ(outputs.NumEntries(), 2);
  const auto &header =
      outputs.Tag("VIDEO_PRESTREAM").packets[0].Get<mediapipe::VideoHeader>();

  EXPECT_EQ(header.width, 3840);
}

TEST(FFMPEGCaptureScreenCalculatorTest, ManualVideoCheck) {
  auto graph_config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          R"pb(
            output_stream: "video_frames"
            node {
              calculator: "FFMPEGCaptureScreenCalculator"
              output_stream: "VIDEO:video_frames"
              output_stream: "VIDEO_PRESTREAM:video_header"
            }
            node {
              calculator: "YUVToImageCalculator"
              input_stream: "YUV_IMAGE:video_frames"
              output_stream: "IMAGE:images"
            }
          )pb");
  std::vector<mediapipe::Packet> output_packets;
  mediapipe::tool::AddVectorSink("images", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));
  // Waits properly via the API until the graph is done.
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(1, output_packets.size());

  const mediapipe::Image &result_image =
      output_packets[0].Get<mediapipe::Image>();

  cv::Mat output_mat =
      mediapipe::formats::MatView(result_image.GetImageFrameSharedPtr().get());
  cv::cvtColor(output_mat, output_mat, cv::COLOR_RGB2BGR);
  ASSERT_TRUE(cv::imwrite("/tmp/image.bmp", output_mat));
}
} // namespace
} // namespace aikit
