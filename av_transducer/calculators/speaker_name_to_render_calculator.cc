

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/util/render_data.pb.h"

namespace aikit {

// Calculator prepares speaker's name to render
//
// Example config:
// node {
//   calculator: "SpeakerNameToRenderCalculator"
//   input_stream: "SPEAKER_NAME:speaker_name"
//   output_stream: "RENDER_DATA:render_data"
// }
class SpeakerNameToRenderCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<std::string> kInSpeakerName{
      "SPEAKER_NAME"};
  static constexpr mediapipe::api2::Output<mediapipe::RenderData>
      kOutRenderData{"RENDER_DATA"};
  MEDIAPIPE_NODE_CONTRACT(kInSpeakerName, kOutRenderData);

  absl::Status Process(mediapipe::CalculatorContext *cc) override;
};
MEDIAPIPE_REGISTER_NODE(SpeakerNameToRenderCalculator);

absl::Status
SpeakerNameToRenderCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &speaker_name = kInSpeakerName(cc).Get();

  ABSL_LOG(INFO) << speaker_name;
  mediapipe::RenderData render_data;
  auto *annotation = render_data.add_render_annotations();

  auto *text = annotation->mutable_text();
  text->set_display_text(absl::StrCat("Speaker: ", speaker_name));
  text->set_left(10);
  text->set_baseline(10);

  kOutRenderData(cc).Send(std::move(render_data));
  return absl::OkStatus();
}

} // namespace aikit
