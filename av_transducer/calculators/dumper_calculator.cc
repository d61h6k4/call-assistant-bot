

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "ml/detection/model.h"
#include <vector>

namespace aikit {

// Dumps input to stdout
//
// Example config:
// node {
//   calculator: "DumperCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "STRING:speaker_name"
// }
class DumperCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<std::vector<ml::Detection>>
      kInDetections{"DETECTIONS"};
  static constexpr mediapipe::api2::Input<std::string>::Optional kInSpeakerName{
      "STRING"};

  MEDIAPIPE_NODE_CONTRACT(kInDetections, kInSpeakerName);

  absl::Status Process(mediapipe::CalculatorContext *cc) override;
};
MEDIAPIPE_REGISTER_NODE(DumperCalculator);

absl::Status DumperCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &detections = kInDetections(cc).Get();

  for (auto d : detections) {
    ABSL_LOG(INFO) << "[" << d.x_center << ", " << d.y_center << ", " << d.width << ", "
                   << d.height << "] " << d.label_id << " " << d.score;
  }

  if (!kInSpeakerName(cc).IsEmpty()) {
    const auto &speaker_name = kInSpeakerName(cc).Get();
    ABSL_LOG(INFO) << "Speaker is " << speaker_name;
  }

  return absl::OkStatus();
}

} // namespace aikit
