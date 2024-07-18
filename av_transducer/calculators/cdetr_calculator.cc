
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/image.h"
#include "ml/detection/model.h"
#include <memory>
#include <vector>

namespace aikit {

// This Calculator applies CDETR object detection
// model to the given frames.
//
// Example config:
// node {
//   calculator: "CDETRCalculator"
//   input_side_packet: "CDETR_MODEL_PATH:cdetr_model_path"
//   input_stream: "IMAGE:image"
//   output_stream: "DETECTIONS:detections"
// }
class CDETRCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInCDETRModelPath{
      "CDETR_MODEL_PATH"};
  static constexpr mediapipe::api2::Input<mediapipe::Image> kInImage{"IMAGE"};
  static constexpr mediapipe::api2::Output<std::vector<ml::Detection>>
      kOutDetections{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInCDETRModelPath, kInImage, kOutDetections);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  std::unique_ptr<ml::CDetr> model_;
};
MEDIAPIPE_REGISTER_NODE(CDETRCalculator);

absl::Status CDETRCalculator::Open(mediapipe::CalculatorContext *cc) {
  const std::string &cdetr_model_path = kInCDETRModelPath(cc).Get();
  model_ = std::make_unique<ml::CDetr>(cdetr_model_path);
  return absl::OkStatus();
}

absl::Status CDETRCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &image = kInImage(cc).Get();
  auto detections =
      model_->operator()(image.GetImageFrameSharedPtr()->PixelData());
  kOutDetections(cc).Send(detections);
  return absl::OkStatus();
}

} // namespace aikit
