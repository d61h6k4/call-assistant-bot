#include "av_transducer/utils/video.h"
#include "libyuv/convert.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/yuv_image.h"

namespace aikit {

// This Calculator converts video frame to YUVImage.
// YUVImage is format of mediapipe, here we connect
// out format with mediapipe one.
//
// Example config:
// node {
//   calculator: "LiftToYUVImageCalculator"
//   input_side_packet: "IN_VIDEO_HEADER:in_video_header"
//   input_stream: "IN_VIDEO:in_video"
//   output_stream: "OUT_VIDEO:out_video"
// }
class LiftToYUVImageCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<media::VideoStreamParameters>
      kInInVideoHeader{"IN_VIDEO_HEADER"};
  static constexpr mediapipe::api2::Input<media::VideoFrame> kInVideo{
      "IN_VIDEO"};
  static constexpr mediapipe::api2::Output<mediapipe::YUVImage> kOutVideo{
      "OUT_VIDEO"};
  MEDIAPIPE_NODE_CONTRACT(
      kInInVideoHeader, kInVideo, kOutVideo);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  media::VideoStreamParameters in_video_stream_parameters_ = {};
};
MEDIAPIPE_REGISTER_NODE(LiftToYUVImageCalculator);

absl::Status LiftToYUVImageCalculator::Open(mediapipe::CalculatorContext *cc) {
  auto in_video_stream_parameters_ = kInInVideoHeader(cc).Get();
  return absl::OkStatus();
}

absl::Status
LiftToYUVImageCalculator::Process(mediapipe::CalculatorContext *cc) {
  const auto &video_frame = kInVideo(cc).Get();

  auto frame = video_frame.c_frame();

  const size_t y_size = frame->linesize[0] * in_video_stream_parameters_.height;
  const size_t u_size =
      frame->linesize[1] * ((in_video_stream_parameters_.height + 1) / 2);
  const size_t v_size =
      frame->linesize[2] * ((in_video_stream_parameters_.height + 1) / 2);

  auto y = std::make_unique<uint8_t[]>(y_size);
  auto u = std::make_unique<uint8_t[]>(u_size);
  auto v = std::make_unique<uint8_t[]>(v_size);

  libyuv::I420Copy(
      frame->data[0], frame->linesize[0], frame->data[1], frame->linesize[1],
      frame->data[2], frame->linesize[2], y.get(), frame->linesize[0], u.get(),
      frame->linesize[1], v.get(), frame->linesize[2],
      in_video_stream_parameters_.width, in_video_stream_parameters_.height);
  mediapipe::api2::PacketAdopting<mediapipe::YUVImage>(
      new mediapipe::YUVImage(
          libyuv::FOURCC_I420, std::move(y), frame->linesize[0], std::move(u),
          frame->linesize[1], std::move(v), frame->linesize[2],
          in_video_stream_parameters_.width,
          in_video_stream_parameters_.height))
      .At(cc->InputTimestamp());

  return absl::OkStatus();
}

} // namespace aikit
