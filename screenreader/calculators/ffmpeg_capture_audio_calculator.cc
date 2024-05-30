#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "screenreader/utils/ffmpeg.h"
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
//   output_stream: "AUDIO:audio_frames"
// }
class FFMPEGCaptureAudioCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Output<std::vector<float>>::Optional
      kOutAudio{"AUDIO"};

  MEDIAPIPE_NODE_CONTRACT(
      kOutAudio, mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  utils::VideoStreamContext video_stream_context_;
  mediapipe::Timestamp prev_audio_timestamp_ = mediapipe::Timestamp::Unset();

  // https://ffmpeg.org/doxygen/trunk/structAVFrame.html
  AVFrame *frame_ = nullptr;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGCaptureAudioCalculator);

absl::Status
FFMPEGCaptureAudioCalculator::Open(mediapipe::CalculatorContext *cc) {

#if __APPLE__
  auto video_stream_context_or = utils::CaptureDevice("avfoundation", ":2");
#elif __linux__
  auto video_stream_context_or = utils::CaptureDevice("pulse", "default");
#endif

  if (!video_stream_context_or.ok()) {
    if (absl::IsFailedPrecondition(video_stream_context_or.status())) {
      ABSL_LOG(WARNING) << "Video stream context was not fully initialized. "
                        << video_stream_context_or.status().message();
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << video_stream_context_or.status().message();
    }
  }

  video_stream_context_ = video_stream_context_or.value();

  if (!video_stream_context_.audio_stream_context.has_value()) {
    return mediapipe::InvalidArgumentError(
        "Video stream does not contain audio stream. Stop processing.");
  }

  frame_ = av_frame_alloc();
  if (!frame_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVFrame";
  }

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);
  av_frame_free(&frame_);

  utils::DestroyVideoStreamContext(video_stream_context_);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureAudioCalculator::Process(mediapipe::CalculatorContext *cc) {
  if (!video_stream_context_.audio_stream_context.has_value()) {
    ABSL_LOG(ERROR) << "There is no audio context. Stop processing.";
    return mediapipe::tool::StatusStop();
  }

  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
  while (av_read_frame(video_stream_context_.format_context, packet_) == 0) {
    if (packet_->stream_index ==
        video_stream_context_.audio_stream_context->stream_index) {

      auto s = utils::PacketToFrame(
          video_stream_context_.audio_stream_context->codec_context, packet_,
          frame_);

      if (s.ok()) {

        // Use microsecond as the unit of time.
        mediapipe::Timestamp timestamp(static_cast<int64_t>(
            static_cast<float>(
                video_stream_context_.audio_stream_context->start_time) +
            video_stream_context_.audio_stream_context->time_base *
                static_cast<float>(frame_->pts) * 1000000.0f));

        // If the timestamp of the current frame is not greater than the one
        // of the previous frame, the new frame will be discarded.
        if (prev_audio_timestamp_ < timestamp) {

          auto audio_data_or = utils::ReadAudioFrame(
              video_stream_context_.audio_stream_context.value(), frame_);
          if (!audio_data_or.ok()) {
            ABSL_LOG(WARNING) << audio_data_or.status().message();
            continue;
          }

          kOutAudio(cc).Send(audio_data_or.value(), timestamp);
          prev_audio_timestamp_ = timestamp;

          // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
          av_packet_unref(packet_);
          return absl::OkStatus();
        } else {
          ABSL_LOG(WARNING) << "Unmonotonic timestamps "
                            << prev_audio_timestamp_ << " and " << timestamp;
        }
      }
    }
    // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
    av_packet_unref(packet_);
  }

  ABSL_LOG(INFO) << "Got last frame";
  return mediapipe::tool::StatusStop();
}

} // namespace aikit
