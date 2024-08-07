

#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "gtest/gtest.h"
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace aikit {
namespace {

class OCRCalculatorTest : public ::testing::Test {
protected:
  OCRCalculatorTest()
      : runner_(R"pb(
                      calculator: "OCRCalculator"
                      input_side_packet: "OCR_MODEL_PATH:ocr_model_path"
                      input_stream: "IMAGE_FRAME:image_frame"
                      output_stream: "STRING:string"
                    )pb") {}

  void SetInput() {
    runner_.MutableSidePackets()->Tag("OCR_MODEL_PATH") =
        mediapipe::MakePacket<std::string>("ml/ocr/models/model.onnx");

    cv::Mat input_mat;
    cv::cvtColor(cv::imread("testdata/participant_name.png"), input_mat,
                 cv::COLOR_BGR2RGB);

    auto input_frame = mediapipe::ImageFrame(
        mediapipe::ImageFormat::SRGB, input_mat.size().width,
        input_mat.size().height);
    input_mat.copyTo(mediapipe::formats::MatView(&input_frame));
    auto input_frame_packet =
        mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));

    auto timestamp = mediapipe::Timestamp(1);

    runner_.MutableInputs()
        ->Tag("IMAGE_FRAME")
        .packets.push_back(input_frame_packet.At(timestamp));
  }

  const std::string &GetOutputs() {
    return runner_.Outputs().Tag("STRING").packets[0].Get<std::string>();
  }

  mediapipe::CalculatorRunner runner_;
};

TEST_F(OCRCalculatorTest, SanityCheck) {
  SetInput();
  MP_ASSERT_OK(runner_.Run());
  auto res = GetOutputs();

  ABSL_LOG(INFO) << res;
}

} // namespace
} // namespace aikit
