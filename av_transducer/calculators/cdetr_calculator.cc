
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/image.h"
#include "ml/detection/model.h"
#include <vector>

namespace aikit {

// This Calculator applies CDETR object detection
// model to the given frames.
//
// Example config:
// node {
//   calculator: "CDETRCalculator"
//   input_stream: "IMAGE:image"
//   output_stream: "DETECTIONS:detections"
// }
class CDETRCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<mediapipe::Image> kInImage{"IMAGE"};
  static constexpr mediapipe::api2::Output<std::vector<ml::Detection>>
      kOutDetections{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInImage, kOutDetections);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  ml::CDetr model_{};
};
MEDIAPIPE_REGISTER_NODE(CDETRCalculator);

absl::Status CDETRCalculator::Open(mediapipe::CalculatorContext *cc) {
  return absl::OkStatus();
}

absl::Status CDETRCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &image = kInImage(cc).Get();
  auto detections = model_(image.GetImageFrameSharedPtr()->PixelData());
  kOutDetections(cc).Send(detections);
  return absl::OkStatus();
}

} // namespace aikit
