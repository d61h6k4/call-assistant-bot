
#include <chrono>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <memory>
#include <vector>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/detection.pb.h"

#include "meeting_bot/evaluator/evaluator.grpc.pb.h"

namespace aikit {

// This Calculator sends all input data to evaluator service.
//
// Example config:
// node {
//   calculator: "EvaluatorClientCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "SPEAKER_NAME:speaker_name"
// }
class EvaluatorClientCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<std::vector<mediapipe::Detection>>
      kInDetections{"DETECTIONS"};
  static constexpr mediapipe::api2::Input<std::string>::Optional kInSpeakerName{
      "SPEAKER_NAME"};
  MEDIAPIPE_NODE_CONTRACT(kInDetections);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<aikit::evaluator::Evaluator::Stub> stub_;
};
MEDIAPIPE_REGISTER_NODE(EvaluatorClientCalculator);

absl::Status EvaluatorClientCalculator::Open(mediapipe::CalculatorContext *cc) {
  channel_ = grpc::CreateChannel("unix:///tmp/evaluator.sock",
                                 grpc::InsecureChannelCredentials());
  stub_ = aikit::evaluator::Evaluator::NewStub(channel_);
  return absl::OkStatus();
}

absl::Status
EvaluatorClientCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &detections = kInDetections(cc).Get();

  grpc::ClientContext context;
  auto deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(100);
  context.set_deadline(deadline);

  aikit::evaluator::DetectionsRequest request;
  request.set_event_timestamp(cc->InputTimestamp().Microseconds());
  for (auto &detection : detections) {

    auto &bbox = detection.location_data().relative_bounding_box();

    auto *d = request.add_detections();
    d->set_xmin(bbox.xmin());
    d->set_ymin(bbox.ymin());
    d->set_width(bbox.width());
    d->set_height(bbox.height());
    d->set_label_id(detection.label_id(0));
    d->set_score(detection.score(0));
  }

  if (!kInSpeakerName(cc).IsEmpty()) {
    const auto &speaker_name = kInSpeakerName(cc).Get();
    request.set_speaker_name(speaker_name);
  }

  aikit::evaluator::DetectionsReply reply;
  auto status = stub_->Detections(&context, request, &reply);

  if (!status.ok()) {
    ABSL_LOG(WARNING) << "Could not send detections to evaluator. "
                      << status.error_message();
  }

  return absl::OkStatus();
}

} // namespace aikit
