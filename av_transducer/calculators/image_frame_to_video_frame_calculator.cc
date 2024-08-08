
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

namespace aikit {

// This Calculator converts ImageFrame to video frame.
//
// Example config:
// node {
//   calculator: "ImageFrameToVideoFrameCalculator"
//   input_stream: "IMAGE_FRAME:in_video"
//   output_side_packet: "OUT_VIDEO_HEADER:out_video_header"
//   output_stream: "VIDEO_FRAME:out_video"
// }
class ImageFrameToVideoFrameCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Input<mediapipe::ImageFrame> kInVideo{
      "IMAGE_FRAME"};
  static constexpr mediapipe::api2::SideOutput<media::VideoStreamParameters>
      kOutVideoHeader{"OUT_VIDEO_HEADER"};
  static constexpr mediapipe::api2::Output<media::VideoFrame> kOutVideo{
      "VIDEO_FRAME"};

  MEDIAPIPE_NODE_CONTRACT(kInVideo, kOutVideoHeader, kOutVideo);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  media::VideoStreamParameters out_video_stream_parameters_{};
};
MEDIAPIPE_REGISTER_NODE(ImageFrameToVideoFrameCalculator);

absl::Status
ImageFrameToVideoFrameCalculator::Open(mediapipe::CalculatorContext *cc) {
  out_video_stream_parameters_.format = AV_PIX_FMT_RGB24;
  kOutVideoHeader(cc).Set(out_video_stream_parameters_);

  return absl::OkStatus();
}

absl::Status
ImageFrameToVideoFrameCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &image_frame = kInVideo(cc).Get();

  auto video_frame = aikit::media::VideoFrame::CreateVideoFrame(
      out_video_stream_parameters_.format, out_video_stream_parameters_.width,
      out_video_stream_parameters_.height);

  auto rgb_frame_mat = ::mediapipe::formats::MatView(&image_frame);
  auto status = video_frame->CopyFromBuffer(rgb_frame_mat.data);
  if (!status.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to create video frame. " << status.message();
  }
  video_frame->SetPTS(cc->InputTimestamp().Microseconds());
  kOutVideo(cc).Send(std::move(video_frame));

  return absl::OkStatus();
}

} // namespace aikit
