#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "screenreader/utils/audio.h"
#include "screenreader/utils/container.h"
#include "screenreader/utils/converter.h"
#include <optional>
#include <vector>

namespace aikit {

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
  static constexpr mediapipe::api2::Output<std::vector<float>> kOutAudio{
      "AUDIO"};
  MEDIAPIPE_NODE_CONTRACT(
      kOutAudioHeader, kOutAudio,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  mediapipe::Timestamp prev_audio_timestamp_ = mediapipe::Timestamp::Unset();
  std::optional<media::ContainerStreamContext> container_stream_context_ =
      std::nullopt;
  std::optional<media::AudioFrame> audio_frame_ = std::nullopt;
  std::optional<media::AudioFrame> out_audio_frame_ = std::nullopt;
  std::optional<media::AudioConverter> audio_converter_ = std::nullopt;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGCaptureAudioCalculator);

absl::Status
FFMPEGCaptureAudioCalculator::Open(mediapipe::CalculatorContext *cc) {

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
  auto out_audio_stream = media::AudioStreamParameters();

  auto out_audio_frame_or = media::AudioFrame::CreateAudioFrame(
      out_audio_stream.format, &out_audio_stream.channel_layout,
      out_audio_stream.sample_rate, out_audio_stream.frame_size);

  if (!out_audio_frame_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVFrame. "
           << out_audio_frame_or.status().message();
  }
  out_audio_frame_ = std::move(out_audio_frame_or.value());

  auto audio_converter_or = media::AudioConverter::CreateAudioConverter(
      in_audio_stream, out_audio_stream);
  if (!audio_converter_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create audio converter. "
           << audio_converter_or.status().message();
  }
  audio_converter_ = std::move(audio_converter_or.value());

  // Write audio header
  kOutAudioHeader(cc).Set(out_audio_stream);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Process(mediapipe::CalculatorContext *cc) {
  auto status = container_stream_context_->ReadPacket(packet_);
  if (status.ok()) {
    status =
        container_stream_context_->PacketToFrame(packet_, audio_frame_.value());
    if (status.ok()) {
      // Use microsecond as the unit of time.
      mediapipe::Timestamp timestamp(
          container_stream_context_->FramePTSInMicroseconds(
              audio_frame_.value()));

      // If the timestamp of the current frame is not greater than the one
      // of the previous frame, the new frame will be discarded.
      if (prev_audio_timestamp_ < timestamp) {

        status = audio_converter_->Convert(audio_frame_.value(),
                                           out_audio_frame_.value());
        if (status.ok()) {

          std::vector<float> copied_audio_data;
          status = out_audio_frame_->AppendAudioData(copied_audio_data);
          if (status.ok()) {
            kOutAudio(cc).Send(copied_audio_data, timestamp);
            prev_audio_timestamp_ = timestamp;

            // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
            av_packet_unref(packet_);
            return absl::OkStatus();

          } else {
            ABSL_LOG(WARNING)
                << "Failed to copy the audio data. " << status.message();
          }
        } else {
          ABSL_LOG(WARNING)
              << "Failed to convert the audio data. " << status.message();
        }
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
