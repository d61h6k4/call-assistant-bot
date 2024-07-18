
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
//   input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
//   input_side_packet: "AUDIO_HEADER:audio_header"
//   input_side_packet: "VIDEO_HEADER:video_header"
//   input_stream: "VIDEO:video_frames"
//   input_stream: "AUDIO:audio_frames"
// }
class FFMPEGSinkVideoCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInFilePath{
      "OUTPUT_FILE_PATH"};
  static constexpr mediapipe::api2::SideInput<media::AudioStreamParameters>
      kInAudioHeader{"AUDIO_HEADER"};
  static constexpr mediapipe::api2::SideInput<media::VideoStreamParameters>
      kInVideoHeader{"VIDEO_HEADER"};
  static constexpr mediapipe::api2::Input<media::VideoFrame>::Optional kInVideo{
      "VIDEO"};
  static constexpr mediapipe::api2::Input<media::AudioFrame>::Optional kInAudio{
      "AUDIO"};

  // TODO(d61h6k4) Check the sinchronization rules, here we may want
  // to use ImmediateInputStreamHandler.
  // https://ai.google.dev/edge/mediapipe/framework/framework_concepts/synchronization
  MEDIAPIPE_NODE_CONTRACT(
      kInFilePath, kInAudioHeader, kInVideoHeader, kInAudio, kInVideo,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  std::optional<media::ContainerStreamContext> container_stream_context_;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *audio_packet_ = nullptr;
  AVPacket *video_packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGSinkVideoCalculator);

absl::Status FFMPEGSinkVideoCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &output_file_path = kInFilePath(cc).Get();
  const auto &audio_stream_parameters = kInAudioHeader(cc).Get();
  const auto &video_stream_parameters = kInVideoHeader(cc).Get();

  auto container_stream_context_or =
      media::ContainerStreamContext::CreateWriterContainerStreamContext(
          audio_stream_parameters, video_stream_parameters, output_file_path);
  if (!container_stream_context_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create container stream context. "
           << container_stream_context_or.status().message();
  }
  container_stream_context_ = std::move(container_stream_context_or.value());

  audio_packet_ = av_packet_alloc();
  if (!audio_packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  video_packet_ = av_packet_alloc();
  if (!video_packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }
  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&audio_packet_);
  av_packet_free(&video_packet_);

  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Process(mediapipe::CalculatorContext *cc) {

  if (kInVideo(cc).IsConnected() && !kInVideo(cc).IsEmpty()) {

    const auto &video_frame = kInVideo(cc).Get();
    auto status =
        container_stream_context_->WriteFrame(video_packet_, &video_frame);
    if (!status.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Failed to write video frame. " << status.message();
    }
  }

  if (kInAudio(cc).IsConnected() && !kInAudio(cc).IsEmpty()) {

    const auto &audio_frame = kInAudio(cc).Get();
    auto status =
        container_stream_context_->WriteFrame(audio_packet_, &audio_frame);
    if (!status.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Failed to write frame. " << status.message();
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
