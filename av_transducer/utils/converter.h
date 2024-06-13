#pragma once

#include "absl/status/statusor.h"
#include "av_transducer/utils/audio.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/audio_fifo.h"
#include "libswresample/swresample.h"
#ifdef __cplusplus
}
#endif

namespace aikit {
namespace media {
class AudioConverter {
public:
  static absl::StatusOr<AudioConverter>
  CreateAudioConverter(const AudioStreamParameters &in_audio_stream,
                       const AudioStreamParameters &out_audio_stream);

  AudioConverter(const AudioConverter &) = delete;
  AudioConverter(AudioConverter &&) noexcept;
  AudioConverter &operator=(const AudioConverter &) = delete;
  AudioConverter &operator=(AudioConverter &&) noexcept;
  ~AudioConverter();

  // Converter works asynchronously, so first you need to store frames
  // and when there is enough data, converter will fill out_frame with
  // converted data.
  absl::Status Store(const AudioFrame *in_frame);
  // Load returns OkStatus, when converter has enough data to fill out
  // frame with converted data, otherwise kFailedPrecondition is returned.
  absl::Status Load(AudioFrame *out_frame);
  // When there is no more input frames, to be sure that we don't have
  // remaining data in converter, call load last frame.
  // You can call it only once.
  absl::Status LoadLastFrame(AudioFrame *out_frame);

private:
  AudioConverter() {};

  absl::Status LoadAlways(AudioFrame *out_frame);

private:
  bool last_frame_was_already_loaded_ = false;

  int in_sample_rate_ = 0;
  int out_sample_rate_ = 0;
  int out_nb_channels_ = 0;
  AVSampleFormat out_sample_format_ = {};
  SwrContext *sw_resample_context_ = nullptr;

  int out_frame_size_ = 0;
  AVAudioFifo *audio_fifo_ = nullptr;
};
} // namespace media
} // namespace aikit
