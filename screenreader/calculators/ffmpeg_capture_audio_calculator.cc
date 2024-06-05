#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "screenreader/utils/audio.h"
#include "screenreader/utils/container.h"
#include "screenreader/utils/converter.h"
#include <csignal>
#include <optional>
namespace aikit {
namespace {
volatile std::sig_atomic_t gSignalStatus;

void SignalHandler(int signal) { gSignalStatus = signal; }
} // namespace
// This Calculator captures audio from audio driver and stream audio packets.
// All streams and input side packets are specified using tags and all of them
// are optional.
//
// Output Streams:
//   AUDIO: Output audio track (floats)
//
// Example config:
// node {
//   calculator: "FFMPEGCaptureAudioCalculator"
//   output_side_packet: "AUDIO_HEADER:audio_header"
//   output_stream: "AUDIO:audio_frames"
// }
class FFMPEGCaptureAudioCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideOutput<media::AudioStreamParameters>
      kOutAudioHeader{"AUDIO_HEADER"};
  static constexpr mediapipe::api2::Output<media::AudioFrame> kOutAudio{
      "AUDIO"};
  MEDIAPIPE_NODE_CONTRACT(
      kOutAudioHeader, kOutAudio,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  mediapipe::Timestamp start_timestamp_ = mediapipe::Timestamp::Unset();
  mediapipe::Timestamp prev_audio_timestamp_ = mediapipe::Timestamp::Unset();
  std::optional<media::ContainerStreamContext> container_stream_context_ =
      std::nullopt;
  std::optional<media::AudioFrame> audio_frame_ = std::nullopt;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGCaptureAudioCalculator);

absl::Status
FFMPEGCaptureAudioCalculator::Open(mediapipe::CalculatorContext *cc) {
  // Register processing system signals
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);

#if __APPLE__
  auto container_stream_context_or =
      media::ContainerStreamContext::CaptureDevice("avfoundation", ":2");
#elif __linux__
  auto container_stream_context_or =
      media::ContainerStreamContext::CaptureDevice("pulse", "default");
#endif

  if (!container_stream_context_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << container_stream_context_or.status().message();
  }
  container_stream_context_ = std::move(container_stream_context_or.value());

  auto audio_frame_or = container_stream_context_->CreateAudioFrame();
  if (!audio_frame_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVFrame. "
           << audio_frame_or.status().message();
  }
  audio_frame_ = std::move(audio_frame_or.value());

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  const auto &in_audio_stream =
      container_stream_context_->GetAudioStreamParameters();

  // Write audio header
  kOutAudioHeader(cc).Set(in_audio_stream);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Process(mediapipe::CalculatorContext *cc) {

  if (gSignalStatus == SIGINT || gSignalStatus == SIGTERM) {
    ABSL_LOG(WARNING) << "Got system singal. Stopping processing.";
    return mediapipe::tool::StatusStop();
  }

  auto status = container_stream_context_->ReadPacket(packet_);
  if (status.ok()) {
    status =
        container_stream_context_->PacketToFrame(packet_, audio_frame_.value());
    if (status.ok()) {

      // Captured frame PTS is current global timestamp in microseconds
      if (start_timestamp_ == mediapipe::Timestamp::Unset()) {
        start_timestamp_ = mediapipe::Timestamp(audio_frame_->c_frame()->pts);
      }

      auto frame_timestamp = mediapipe::Timestamp(audio_frame_->c_frame()->pts);
      container_stream_context_->SetFramePTS(
          (frame_timestamp - start_timestamp_).Microseconds(),
          audio_frame_.value());

      // Use microsecond as the unit of time.
      mediapipe::Timestamp timestamp(
          container_stream_context_->FramePTSInMicroseconds(
              audio_frame_.value()));

      // If the timestamp of the current frame is not greater than the one
      // of the previous frame, the new frame will be discarded.
      if (prev_audio_timestamp_ < timestamp) {
        kOutAudio(cc).Send(std::move(audio_frame_.value()), timestamp);

      } else {
        ABSL_LOG(WARNING) << "Unmonotonic timestamps " << prev_audio_timestamp_
                          << " and " << timestamp;
      }
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "failed to decode a packet. " << status.message();
    }

  } else {
    ABSL_LOG(INFO) << "Failed to read a packet. " << status.message();
  }
  // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
  av_packet_unref(packet_);

  ABSL_LOG(INFO) << "Got last frame";
  return mediapipe::tool::StatusStop();
}

} // namespace aikit
