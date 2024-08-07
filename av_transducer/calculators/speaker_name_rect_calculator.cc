
#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include "ml/formats/detection.h"

namespace aikit {

// Calculator extracts rectangle with speaker's name
//
// Example config:
// node {
//   calculator: "SpeakerNameRectCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "NORM_RECT:norm_rect"
// }
class SpeakerNameRectCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<std::vector<ml::Detection>>
      kInDetections{"DETECTIONS"};
  static constexpr mediapipe::api2::Output<mediapipe::NormalizedRect> kOutRect{
      "NORM_RECT"};
  MEDIAPIPE_NODE_CONTRACT(kInDetections, kOutRect);

  absl::Status Process(mediapipe::CalculatorContext *cc) override;
};
MEDIAPIPE_REGISTER_NODE(SpeakerNameRectCalculator);

absl::Status
SpeakerNameRectCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &detections = kInDetections(cc).Get();

  for (auto &detection : detections) {
    // speaker, only one
    if (detection.label_id == 0) {
      mediapipe::NormalizedRect speaker_name_rect;

      speaker_name_rect.set_x_center(detection.xmin +
                                     detection.width * 0.75f / 2.0f);
      speaker_name_rect.set_y_center(detection.ymin + 0.9 * detection.height);
      speaker_name_rect.set_height(0.2 * detection.height);
      speaker_name_rect.set_width(detection.width * 0.75f);

      kOutRect(cc).Send(speaker_name_rect);
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
