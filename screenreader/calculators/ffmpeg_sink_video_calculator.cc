
#include "mediapipe/framework/api2/node.h"
#include "screenreader/utils/audio.h"
#include "screenreader/utils/container.h"
#include "screenreader/utils/converter.h"
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
//   input_stream: "YUV_IMAGE:video_frames"
//   input_stream: "AUDIO:audio_frames"
// }
class FFMPEGSinkVideoCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInFilePath{
      "OUTPUT_FILE_PATH"};
  static constexpr mediapipe::api2::SideInput<media::AudioStreamParameters>
      kInAudioHeader{"AUDIO_HEADER"};
  // static constexpr mediapipe::api2::Input<mediapipe::YUVImage>::Optional
  //     kInVideo{"YUV_IMAGE"};
  static constexpr mediapipe::api2::Input<media::AudioFrame>::Optional kInAudio{
      "AUDIO"};

  MEDIAPIPE_NODE_CONTRACT(kInFilePath, kInAudioHeader, kInAudio);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  std::optional<media::ContainerStreamContext> container_stream_context_;

  // AAC encoder support only FLTP format
  std::unique_ptr<media::AudioFrame> write_audio_frame_ = nullptr;
  std::optional<media::AudioConverter> audio_converter_ = std::nullopt;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;

  int pts_ = 0;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGSinkVideoCalculator);

absl::Status FFMPEGSinkVideoCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &output_file_path = kInFilePath(cc).Get();
  const auto &audio_stream_parameters = kInAudioHeader(cc).Get();

  auto write_audio_stream_parameters = media::AudioStreamParameters();
  write_audio_stream_parameters.bit_rate = audio_stream_parameters.bit_rate;
  write_audio_stream_parameters.sample_rate =
      audio_stream_parameters.sample_rate;
  write_audio_stream_parameters.frame_size =
      1024; // audio_stream_parameters.frame_size;

  auto container_stream_context_or =
      media::ContainerStreamContext::CreateWriterContainerStreamContext(
          write_audio_stream_parameters, output_file_path);
  if (!container_stream_context_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create container stream context";
  }
  container_stream_context_ = std::move(container_stream_context_or.value());

  write_audio_frame_ = container_stream_context_->CreateAudioFrame();
  if (!write_audio_frame_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVFrame. ";
  }

  auto audio_converter_or = media::AudioConverter::CreateAudioConverter(
      audio_stream_parameters, write_audio_stream_parameters);
  if (!audio_converter_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to create audio converter. "
           << audio_converter_or.status().message();
  }
  audio_converter_ = std::move(audio_converter_or.value());

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }
  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);

  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Process(mediapipe::CalculatorContext *cc) {

  // if (kInVideo(cc).IsConnected() && !kInVideo(cc).IsEmpty()) {
  //   if (video_stream_.has_value()) {

  //     const auto &yuv_image = kInVideo(cc).Get();
  //     libyuv::I420Copy(
  //         yuv_image.data(0), yuv_image.stride(0), yuv_image.data(1),
  //         yuv_image.stride(1), yuv_image.data(2), yuv_image.stride(2),
  //         video_stream_->frame->data[0], video_stream_->frame->linesize[0],
  //         video_stream_->frame->data[1], video_stream_->frame->linesize[1],
  //         video_stream_->frame->data[2], video_stream_->frame->linesize[2],
  //         video_stream_->enc->width, video_stream_->enc->height);
  //     video_stream_->frame->pts = cc->InputTimestamp().Microseconds();

  //     auto res = WriteFrame(output_media_context_, video_stream_->enc,
  //                           video_stream_->st, video_stream_->frame,
  //                           video_stream_->tmp_pkt);
  //     if (!res.ok()) {
  //       return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
  //              << res.message();
  //     }
  //   } else {
  //     ABSL_LOG_FIRST_N(WARNING, 1)
  //         << "Could not find video stream. Based on the ouput file path "
  //            "extension FFMpeg assumes user wants to store only audio stream,
  //            " "Please rename output file path (e.g. use .mp4 ext) if you
  //            want " "to save video stream too.";
  //   }
  // }

  if (kInAudio(cc).IsConnected() && !kInAudio(cc).IsEmpty()) {

    const auto &audio_frame = kInAudio(cc).Get();

    auto status = audio_converter_->Store(&audio_frame);
    if (!status.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Failed to convert frame. " << status.message();
    }
    status = audio_converter_->Load(write_audio_frame_.get());
    if (status.ok()) {
      write_audio_frame_->SetPTS(pts_);
      pts_ = pts_ + write_audio_frame_->c_frame()->nb_samples;

      status = container_stream_context_->WriteFrame(packet_,
                                                     write_audio_frame_.get());
      if (!status.ok()) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "Failed to write frame. " << status.message();
      }
    } else if (!absl::IsFailedPrecondition(status)) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Failed to resample frame. " << status.message();
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
