

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/container.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/node.h"
#include <csignal>
#include <optional>

namespace aikit {

// Calculator takes video (images) stream (optional) and audio stream
// (optional), muxes them and writes to a file.
//
// Example config:
// node {
//   calculator: "FFMPEGSinkVideoCalculator"
//   input_side_packet: "INPUT_FILE_PATH:input_file_path"
//   output_side_packet: "AUDIO_HEADER:audio_header"
//   output_side_packet: "VIDEO_HEADER:video_header"
//   ouput_stream: "VIDEO:video_frames"
//   output_stream: "AUDIO:audio_frames"
// }
class FFMPEGSourceVideoCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInFilePath{
      "INPUT_FILE_PATH"};
  static constexpr mediapipe::api2::SideOutput<media::AudioStreamParameters>
      kOutAudioHeader{"AUDIO_HEADER"};
  static constexpr mediapipe::api2::SideOutput<media::VideoStreamParameters>
      kOutVideoHeader{"VIDEO_HEADER"};
  static constexpr mediapipe::api2::Output<media::VideoFrame>::Optional
      kOutVideo{"VIDEO"};
  static constexpr mediapipe::api2::Output<media::AudioFrame>::Optional
      kOutAudio{"AUDIO"};

  MEDIAPIPE_NODE_CONTRACT(
      kInFilePath, kOutAudioHeader, kOutVideoHeader, kOutAudio, kOutVideo,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  std::optional<media::ContainerStreamContext> container_stream_context_;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGSourceVideoCalculator);

absl::Status
FFMPEGSourceVideoCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &input_file_path = kInFilePath(cc).Get();

  auto container_stream_context_or =
      media::ContainerStreamContext::CreateReaderContainerStreamContext(
          input_file_path, nullptr);
  if (!container_stream_context_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create container stream context. "
           << container_stream_context_or.status().message();
  }
  container_stream_context_ = std::move(container_stream_context_or.value());

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  kOutAudioHeader(cc).Set(
      container_stream_context_->GetAudioStreamParameters());
  kOutVideoHeader(cc).Set(
      container_stream_context_->GetVideoStreamParameters());
  return absl::OkStatus();
}

absl::Status
FFMPEGSourceVideoCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);

  return absl::OkStatus();
}

absl::Status
FFMPEGSourceVideoCalculator::Process(mediapipe::CalculatorContext *cc) {

  absl::Status status = container_stream_context_->ReadPacket(packet_);

  if (absl::IsAborted(status)) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to read packet. " << status.message();
  } else if (absl::IsFailedPrecondition(status)) {
    return mediapipe::tool::StatusStop();
  }

  if (container_stream_context_->IsPacketAudio(packet_)) {

    auto audio_frame_or = container_stream_context_->CreateAudioFrame();
    status =
        container_stream_context_->PacketToFrame(packet_, audio_frame_or.get());
    if (status.ok()) {
      kOutAudio(cc).Send(std::move(audio_frame_or));
      return absl::OkStatus();
    }
    // else {
    //   return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
    //          << "failed to decode an audio packet. " << status.message();
    // }
  } else if (container_stream_context_->IsPacketVideo(packet_)) {

    auto video_frame_or = container_stream_context_->CreateVideoFrame();
    status =
        container_stream_context_->PacketToFrame(packet_, video_frame_or.get());
    if (status.ok()) {
      kOutVideo(cc).Send(std::move(video_frame_or));
      return absl::OkStatus();
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "failed to decode a video packet. " << status.message();
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
