

#include "absl/log/absl_log.h"
#include "av_transducer/utils/video.h"
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

class ImageFrameToVideoFrameCalculatorTest : public ::testing::Test {
protected:
  ImageFrameToVideoFrameCalculatorTest()
      : runner_(R"pb(
            calculator: "ImageFrameToVideoFrameCalculator"
            input_stream: "IMAGE_FRAME:in_video"
            output_side_packet: "OUT_VIDEO_HEADER:out_video_header"
            output_stream: "VIDEO_FRAME:out_video"
        )pb") {}

  void SetInput() {
    cv::Mat input_mat;
    cv::cvtColor(cv::imread("testdata/meeting_frame.png"), input_mat,
                 cv::COLOR_BGR2RGB);

    auto input_frame =
        mediapipe::ImageFrame(mediapipe::ImageFormat::SRGB,
                              input_mat.size().width, input_mat.size().height);
    input_mat.copyTo(mediapipe::formats::MatView(&input_frame));
    auto input_frame_packet =
        mediapipe::MakePacket<mediapipe::ImageFrame>(std::move(input_frame));

    auto timestamp = mediapipe::Timestamp(1);

    runner_.MutableInputs()
        ->Tag("IMAGE_FRAME")
        .packets.push_back(input_frame_packet.At(timestamp));
  }

  const media::VideoFrame &GetOutputs() {
    return runner_.Outputs()
        .Tag("VIDEO_FRAME")
        .packets[0]
        .Get<media::VideoFrame>();
  }

  mediapipe::CalculatorRunner runner_;
};

TEST_F(ImageFrameToVideoFrameCalculatorTest, SanityCheck) {
  SetInput();
  MP_ASSERT_OK(runner_.Run());
  auto& res = GetOutputs();

  ABSL_LOG(INFO) << res.GetPTS();
}

} // namespace
} // namespace aikit
