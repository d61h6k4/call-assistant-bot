

#include <vector>
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "ml/detection/model.h"

namespace aikit {

// Dumps input to stdout
//
// Example config:
// node {
//   calculator: "DumperCalculator"
//   input_stream: "DETECTIONS:detections"
// }
class DumperCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<std::vector<ml::Detection>>
      kInDetections{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDetections);

  absl::Status Process(mediapipe::CalculatorContext *cc) override;

};
MEDIAPIPE_REGISTER_NODE(DumperCalculator);

absl::Status DumperCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &detections = kInDetections(cc).Get();

  for (auto d : detections) {
      ABSL_LOG(INFO) << "[" << d.xmin << ", " << d.ymin << ", " << d.width << ", "
                     << d.height << "] " << d.label_id << " " << d.score;
    }
  return absl::OkStatus();
}

} // namespace aikit
