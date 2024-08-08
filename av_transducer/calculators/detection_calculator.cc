
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "ml/detection/model.h"
#include <memory>
#include <vector>

namespace aikit {

namespace {
// Needs to be coherent with ml/detection/train.py:L26
std::string LabelId2Label(int label_id) {
  switch (label_id) {
  case 0:
    return "speaker";
  case 1:
    return "participant";
  case 2:
    return "shared screen";
  case 3:
    return "black screen";
  case 4:
    return "welcome page";
  case 5:
    return "alone";
  case 6:
    return "name";
  }

  return "unk";
}

} // namespace

// This Calculator applies object detection
// model to the given frames.
//
// Example config:
// node {
//   calculator: "DetectionCalculator"
//   input_side_packet: "MODEL_PATH:model_path"
//   input_stream: "IMAGE:image"
//   output_stream: "DETECTIONS:detections"
// }
class DetectionCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string>
      kInDetectionModelPath{"MODEL_PATH"};
  static constexpr mediapipe::api2::Input<mediapipe::Image> kInImage{"IMAGE"};
  static constexpr mediapipe::api2::Output<std::vector<mediapipe::Detection>>
      kOutDetections{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInDetectionModelPath, kInImage, kOutDetections);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  std::unique_ptr<ml::CDetr> model_;
};
MEDIAPIPE_REGISTER_NODE(DetectionCalculator);

absl::Status DetectionCalculator::Open(mediapipe::CalculatorContext *cc) {
  const std::string &cdetr_model_path = kInDetectionModelPath(cc).Get();
  model_ = std::make_unique<ml::CDetr>(cdetr_model_path);
  return absl::OkStatus();
}

absl::Status DetectionCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &image = kInImage(cc).Get();
  auto detections =
      model_->operator()(image.GetImageFrameSharedPtr()->PixelData());

  std::vector<mediapipe::Detection> mdets;
  for (const auto &detection : detections) {
    mediapipe::Detection mdet;
    mdet.mutable_label()->Add(LabelId2Label(detection.label_id));
    mdet.mutable_label_id()->Add(detection.label_id);
    mdet.mutable_score()->Add(detection.score);

    auto *location_data = mdet.mutable_location_data();
    location_data->set_format(::mediapipe::LocationData_Format::
                                  LocationData_Format_RELATIVE_BOUNDING_BOX);
    auto *bbox = location_data->mutable_relative_bounding_box();
    bbox->set_xmin(detection.x_center - detection.width * 0.5f);
    bbox->set_ymin(detection.y_center - detection.height * 0.5f);
    bbox->set_width(detection.width);
    bbox->set_height(detection.height);

    mdets.emplace_back(std::move(mdet));
  }

  kOutDetections(cc).Send(mdets);

  return absl::OkStatus();
}

} // namespace aikit
