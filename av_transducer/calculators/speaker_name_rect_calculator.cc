
#include <optional>
#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

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
  static constexpr mediapipe::api2::Input<std::vector<mediapipe::Detection>>
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
    if (detection.label_id(0) == 0) {
      auto &speaker_rect = detection.location_data().relative_bounding_box();
      for (auto &detection : detections) {
        if (detection.label_id(0) == 6) {
          auto &bbox = detection.location_data().relative_bounding_box();
          // Check name inside speaker's rectangle
          if (speaker_rect.xmin() <= bbox.xmin() &&
              speaker_rect.ymin() <= bbox.ymin() &&
              bbox.xmin() + bbox.width() <
                  speaker_rect.xmin() + speaker_rect.width() &&
              bbox.ymin() + bbox.height() <
                  speaker_rect.ymin() + speaker_rect.height()) {

            mediapipe::NormalizedRect speaker_name_rect;

            speaker_name_rect.set_x_center(bbox.xmin() + bbox.width() / 2.0f);
            speaker_name_rect.set_y_center(bbox.ymin() + bbox.height() / 2.0f);
            speaker_name_rect.set_height(bbox.height());
            speaker_name_rect.set_width(bbox.width());

            kOutRect(cc).Send(speaker_name_rect);
            return absl::OkStatus();
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
