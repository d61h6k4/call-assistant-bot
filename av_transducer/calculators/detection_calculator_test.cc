
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "ml/detection/model.h"
#include "gtest/gtest.h"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace aikit {
namespace {

class CDetrCalculatorTest : public ::testing::Test {
protected:
  CDetrCalculatorTest()
      : runner_(R"pb(
                      calculator: "DetectionCalculator"
                      input_side_packet: "MODEL_PATH:model_path"
                      input_stream: "IMAGE:image"
                      output_stream: "DETECTIONS:detections"
                    )pb") {}

  void SetInput() {
    runner_.MutableSidePackets()->Tag("MODEL_PATH") =
        mediapipe::MakePacket<std::string>("ml/detection/models/model.onnx");

    cv::Mat input_mat;
    cv::cvtColor(cv::imread("testdata/meeting_frame.png"), input_mat,
                 cv::COLOR_BGR2RGB);
    auto input_frame = std::make_shared<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, input_mat.size().width,
        input_mat.size().height);
    input_mat.copyTo(mediapipe::formats::MatView(input_frame.get()));
    auto input_frame_packet =
        mediapipe::MakePacket<mediapipe::Image>(std::move(input_frame));

    auto timestamp = mediapipe::Timestamp(0);

    runner_.MutableInputs()->Tag("IMAGE").packets.push_back(
        input_frame_packet.At(timestamp));
  }

  const std::vector<mediapipe::Detection> &GetOutputs() {
    return runner_.Outputs()
        .Tag("DETECTIONS")
        .packets[0]
        .Get<std::vector<mediapipe::Detection>>();
  }

  mediapipe::CalculatorRunner runner_;
};

TEST_F(CDetrCalculatorTest, SanityCheck) {
  SetInput();
  MP_ASSERT_OK(runner_.Run());
  auto &det = GetOutputs();

  EXPECT_EQ(det.size(), 10);
  for (const auto& d : det) {
    ABSL_LOG(INFO) << d.DebugString() << "\n";
  }
}

} // namespace
} // namespace aikit
