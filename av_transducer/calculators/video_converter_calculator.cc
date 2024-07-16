

#include "av_transducer/utils/converter.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include <optional>
namespace aikit {

// This Calculator converts video frame from one format to another.
//
// Example config:
// node {
//   calculator: "VideoConverterCalculator"
//   input_side_packet: "IN_VIDEO_HEADER:in_video_header"
//   input_side_packet: "OUT_VIDEO_HEADER:out_video_header"
//   input_stream: "IN_VIDEO:in_video"
//   output_stream: "OUT_VIDEO:out_video"
// }
class VideoConverterCalculator: public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<media::VideoStreamParameters>
      kInInVideoHeader{"IN_VIDEO_HEADER"};
  static constexpr mediapipe::api2::SideInput<media::VideoStreamParameters>
      kInOutVideoHeader{"OUT_VIDEO_HEADER"};
  static constexpr mediapipe::api2::Input<media::VideoFrame> kInVideo{
      "IN_VIDEO"};
  static constexpr mediapipe::api2::Output<media::VideoFrame> kOutVideo{
      "OUT_VIDEO"};
  MEDIAPIPE_NODE_CONTRACT(
      kInInVideoHeader, kInOutVideoHeader, kInVideo, kOutVideo,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  std::optional<media::VideoConverter> video_converter_ = std::nullopt;
  media::VideoStreamParameters out_video_stream_parameters_ = {};
};
MEDIAPIPE_REGISTER_NODE(VideoConverterCalculator);

absl::Status VideoConverterCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &in_video_stream_parameters = kInInVideoHeader(cc).Get();
  const auto &out_video_stream_parameters = kInOutVideoHeader(cc).Get();

  auto video_converter_or = aikit::media::VideoConverter::CreateVideoConverter(
      in_video_stream_parameters, out_video_stream_parameters);

  if (!video_converter_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << video_converter_or.status().message();
  }
  video_converter_ = std::move(video_converter_or.value());

  out_video_stream_parameters_ = out_video_stream_parameters;

  return absl::OkStatus();
}

absl::Status
VideoConverterCalculator::Process(mediapipe::CalculatorContext *cc) {

  const auto &video_frame = kInVideo(cc).Get();

  auto write_video_frame = media::VideoFrame::CreateVideoFrame(
      out_video_stream_parameters_.format, out_video_stream_parameters_.width,
      out_video_stream_parameters_.height);
  if (!write_video_frame) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create video frame.";
  }

  auto status =
      video_converter_->Convert(&video_frame, write_video_frame.get());
  if (!status.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to convert frame. " << status.message();
  }
  kOutVideo(cc).Send(std::move(write_video_frame));
  return absl::OkStatus();
}

} // namespace aikit
