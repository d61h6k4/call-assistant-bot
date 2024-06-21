#include "av_transducer/utils/container.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include <csignal>
#include <optional>

namespace aikit {
namespace {
volatile std::sig_atomic_t gSignalStatus;

void SignalHandler(int signal) { gSignalStatus = signal; }
} // namespace
// This Calculator captures screen and produces video packets.
//
// Example config:
// node {
//   calculator: "FFMPEGCaptureScreenCalculator"
//   output_stream: "VIDEO:video_frames"
//   output_side_packet: "VIDEO_HEADER:video_header"
// }
class FFMPEGCaptureScreenCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Output<media::VideoFrame> kOutVideo{
      "VIDEO"};
  static constexpr mediapipe::api2::SideOutput<media::VideoStreamParameters>
      kOutVideoHeader{"VIDEO_HEADER"};

  MEDIAPIPE_NODE_CONTRACT(
      kOutVideo, kOutVideoHeader,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  mediapipe::Timestamp start_timestamp_ = mediapipe::Timestamp::Unset();
  mediapipe::Timestamp prev_video_timestamp_ = mediapipe::Timestamp::Unset();

  std::optional<media::ContainerStreamContext> container_stream_context_ =
      std::nullopt;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGCaptureScreenCalculator);

absl::Status
FFMPEGCaptureScreenCalculator::Open(mediapipe::CalculatorContext *cc) {

#if __APPLE__
  auto container_stream_context_or =
      media::ContainerStreamContext::CaptureDevice("avfoundation", "3:");
#elif __linux__
  // :1 is DISPLAY value
  auto container_stream_context_or =
      media::ContainerStreamContext::CaptureDevice("x11grab", ":1.0");
#endif

  if (!container_stream_context_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << container_stream_context_or.status().message();
  }

  container_stream_context_ = std::move(container_stream_context_or.value());

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  const auto &in_video_stream =
      container_stream_context_->GetVideoStreamParameters();

  // Write video header
  kOutVideoHeader(cc).Set(in_video_stream);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureScreenCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);

  ABSL_LOG(INFO) << "Screen reader close";
  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureScreenCalculator::Process(mediapipe::CalculatorContext *cc) {
  if (gSignalStatus == SIGINT || gSignalStatus == SIGTERM) {
    ABSL_LOG(WARNING) << "Got system singal. Stopping video processing.";
    return mediapipe::tool::StatusStop();
  }

  auto status = container_stream_context_->ReadPacket(packet_);
  if (status.ok()) {

    auto video_frame_or = container_stream_context_->CreateVideoFrame();
    if (!video_frame_or) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "failed to allocate memory for Video Frame ";
    }

    status =
        container_stream_context_->PacketToFrame(packet_, video_frame_or.get());
    if (status.ok()) {
      // Captured frame PTS is current global timestamp in microseconds
      if (start_timestamp_ == mediapipe::Timestamp::Unset()) {
        start_timestamp_ = mediapipe::Timestamp(video_frame_or->GetPTS());
      }

      auto current_timestamp = mediapipe::Timestamp(video_frame_or->GetPTS());
      // x11grab time base is 1/0, so we set pts manually
      video_frame_or->SetPTS(
          av_rescale_q((current_timestamp - start_timestamp_).Microseconds(),
                       AVRational{1, 1000000}, AVRational{1, 30}));
      auto timestamp = mediapipe::Timestamp(av_rescale_q(
          video_frame_or->GetPTS(), AVRational{1, 30}, AVRational{1, 1000000}));
      // If the timestamp of the current frame is not greater than the one
      // of the previous frame, the new frame will be discarded.
      if (prev_video_timestamp_ < timestamp) {
        kOutVideo(cc).Send(std::move(video_frame_or), timestamp);
        prev_video_timestamp_ = timestamp;

        // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
        av_packet_unref(packet_);
        return absl::OkStatus();
      } else {
        ABSL_LOG_EVERY_N_SEC(WARNING, 3)
            << "Unmonotonic timestamps " << prev_video_timestamp_ << " and "
            << timestamp << " PTS: " << video_frame_or->GetPTS() << " diff: "
            << (current_timestamp - start_timestamp_).Microseconds();
        av_packet_unref(packet_);
        return absl::OkStatus();
      }
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "failed to decode a video packet. " << status.message();
    }

  } else {
    ABSL_LOG(INFO) << "Failed to read a video packet. " << status.message();
  }

  // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
  av_packet_unref(packet_);

  ABSL_LOG(INFO) << "Got last video frame";
  return mediapipe::tool::StatusStop();
}

} // namespace aikit
