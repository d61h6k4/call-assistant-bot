#pragma once

#include "absl/status/statusor.h"
#include "screenreader/utils/audio.h"

#ifdef __cplusplus
extern "C" {
#endif
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

  absl::Status Convert(const AudioFrame &in_frame, AudioFrame &out_frame);

private:
  AudioConverter() {};

private:
  int in_sample_rate_ = 0;
  int out_sample_rate_ = 0;
  SwrContext *sw_resample_context_ = nullptr;
};
} // namespace media
} // namespace aikit
