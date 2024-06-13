
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/converter.h"
#include <optional>
namespace aikit {

// This Calculator converts audio frame from one format to another.
//
// Example config:
// node {
//   calculator: "AudioConverterCalculator"
//   input_side_packet: "IN_AUDIO_HEADER:in_audio_header"
//   input_side_packet: "OUT_AUDIO_HEADER:out_audio_header"
//   input_stream: "IN_AUDIO:in_audio"
//   output_stream: "OUT_AUDIO:out_audio"
// }
class AudioConverterCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<media::AudioStreamParameters>
      kInInAudioHeader{"IN_AUDIO_HEADER"};
  static constexpr mediapipe::api2::SideInput<media::AudioStreamParameters>
      kInOutAudioHeader{"OUT_AUDIO_HEADER"};
  static constexpr mediapipe::api2::Input<media::AudioFrame> kInAudio{
      "IN_AUDIO"};
  static constexpr mediapipe::api2::Output<media::AudioFrame> kOutAudio{
      "OUT_AUDIO"};
  MEDIAPIPE_NODE_CONTRACT(
      kInInAudioHeader, kInOutAudioHeader, kInAudio, kOutAudio,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
  // Output audio frame pts is how many samples in total in all audio track,
  // we collect number of output samples in this variable.
  // Timestamp of an output audio frame is based on the PTS.
  int out_pts_ = 0;
  mediapipe::Timestamp prev_audio_timestamp_ = mediapipe::Timestamp::Unset();
  std::optional<media::AudioConverter> audio_converter_ = std::nullopt;
  media::AudioStreamParameters out_audio_stream_parameters_ = {};
};
MEDIAPIPE_REGISTER_NODE(AudioConverterCalculator);

absl::Status AudioConverterCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &in_audio_stream_parameters = kInInAudioHeader(cc).Get();
  const auto &out_audio_stream_parameters = kInOutAudioHeader(cc).Get();

  auto audio_converter_or = aikit::media::AudioConverter::CreateAudioConverter(
      in_audio_stream_parameters, out_audio_stream_parameters);

  if (!audio_converter_or.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << audio_converter_or.status().message();
  }
  audio_converter_ = std::move(audio_converter_or.value());

  out_audio_stream_parameters_ = out_audio_stream_parameters;

  return absl::OkStatus();
}

absl::Status
AudioConverterCalculator::Process(mediapipe::CalculatorContext *cc) {

  const auto &audio_frame = kInAudio(cc).Get();
  auto status = audio_converter_->Store(&audio_frame);
  if (!status.ok()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to convert frame. " << status.message();
  }

  while (status.ok()) {
    auto write_audio_frame = media::AudioFrame::CreateAudioFrame(
        out_audio_stream_parameters_.format,
        &out_audio_stream_parameters_.channel_layout,
        out_audio_stream_parameters_.sample_rate,
        out_audio_stream_parameters_.frame_size);
    status = audio_converter_->Load(write_audio_frame.get());

    if (status.ok()) {
      write_audio_frame->SetPTS(out_pts_);
      out_pts_ = out_pts_ + write_audio_frame->c_frame()->nb_samples;

      // PTS - number of samples in audio track
      // 1/sample_rate - how many samples per second, so PTS * 1 / sample_rate -
      // how many seconds we in autio track
      // seconds * 1000000 = microseconds
      auto timestamp = mediapipe::Timestamp(
          av_rescale_q(write_audio_frame->GetPTS(),
                       AVRational{1, out_audio_stream_parameters_.sample_rate},
                       AVRational{1, 1000000}));

      // If the timestamp of the current frame is not greater than the one
      // of the previous frame, the new frame will be discarded.
      if (prev_audio_timestamp_ < timestamp) {
        kOutAudio(cc).Send(std::move(write_audio_frame), timestamp);
        prev_audio_timestamp_ = timestamp;
      } else {
        ABSL_LOG(WARNING) << "Unmonotonic timestamps " << prev_audio_timestamp_
                          << " and " << timestamp;
      }
    } else if (absl::IsAborted(status)) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Could not convert frame. " << status.message();
    }
  }
  return absl::OkStatus();
}

} // namespace aikit
