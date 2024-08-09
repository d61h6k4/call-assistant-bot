
#include <memory>
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/ocr/model.h"

namespace aikit {

// This Calculator applies OCR
// model to the given frames.
//
// Example config:
// node {
//   calculator: "OCRCalculator"
//   input_side_packet: "OCR_MODEL_PATH:ocr_model_path"
//   input_stream: "IMAGE_FRAME:image_frame"
//   output_stream: "STRING:string"
// }
class OCRCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInOCRModelPath{
      "OCR_MODEL_PATH"};
  static constexpr mediapipe::api2::Input<mediapipe::ImageFrame> kInImage{
      "IMAGE_FRAME"};
  static constexpr mediapipe::api2::Output<std::string> kOutDetections{
      "STRING"};
  MEDIAPIPE_NODE_CONTRACT(kInOCRModelPath, kInImage, kOutDetections);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  std::unique_ptr<ml::OCR> model_;
};
MEDIAPIPE_REGISTER_NODE(OCRCalculator);

absl::Status OCRCalculator::Open(mediapipe::CalculatorContext *cc) {
  const std::string &model_path = kInOCRModelPath(cc).Get();
  model_ = std::make_unique<ml::OCR>(model_path);

  return absl::OkStatus();
}

absl::Status OCRCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &rgb_frame = kInImage(cc).Get();

  mediapipe::ImageFrame gray_frame(mediapipe::ImageFormat::GRAY8,
                                   rgb_frame.Width(), rgb_frame.Height());
  cv::Mat rgb_frame_mat = ::mediapipe::formats::MatView(&rgb_frame);
  cv::Mat gray_frame_mat = ::mediapipe::formats::MatView(&gray_frame);
  cv::cvtColor(rgb_frame_mat, gray_frame_mat, CV_RGB2GRAY);

  auto text = model_->operator()(gray_frame_mat.data);
  kOutDetections(cc).Send(text);
  return absl::OkStatus();
}

} // namespace aikit
