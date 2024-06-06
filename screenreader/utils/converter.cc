
#include "screenreader/utils/audio.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/opt.h"
#ifdef __cplusplus
}
#endif

#include "screenreader/utils/converter.h"

namespace aikit {
namespace media {

absl::StatusOr<AudioConverter> AudioConverter::CreateAudioConverter(
    const AudioStreamParameters &in_audio_stream,
    const AudioStreamParameters &out_audio_stream) {
  AudioConverter audio_converter;

  audio_converter.sw_resample_context_ = swr_alloc();
  if (!audio_converter.sw_resample_context_) {
    return absl::AbortedError("Could not allocated resampler context.");
  }

  av_opt_set_chlayout(audio_converter.sw_resample_context_, "in_chlayout",
                      &in_audio_stream.channel_layout, 0);
  av_opt_set_int(audio_converter.sw_resample_context_, "in_sample_rate",
                 in_audio_stream.sample_rate, 0);
  av_opt_set_sample_fmt(audio_converter.sw_resample_context_, "in_sample_fmt",
                        in_audio_stream.format, 0);
  av_opt_set_chlayout(audio_converter.sw_resample_context_, "out_chlayout",
                      &out_audio_stream.channel_layout, 0);
  av_opt_set_int(audio_converter.sw_resample_context_, "out_sample_rate",
                 out_audio_stream.sample_rate, 0);
  av_opt_set_sample_fmt(audio_converter.sw_resample_context_, "out_sample_fmt",
                        out_audio_stream.format, 0);

  if (int ret = swr_init(audio_converter.sw_resample_context_); ret < 0) {
    return absl::AbortedError("Failed to initialize the resampling context.");
  }

  audio_converter.in_sample_rate_ = in_audio_stream.sample_rate;
  audio_converter.out_sample_rate_ = out_audio_stream.sample_rate;

  return audio_converter;
}

AudioConverter::AudioConverter(AudioConverter &&o) noexcept
    : in_sample_rate_(o.in_sample_rate_), out_sample_rate_(o.out_sample_rate_),
      sw_resample_context_(o.sw_resample_context_) {
  o.sw_resample_context_ = nullptr;
}

AudioConverter &AudioConverter::operator=(AudioConverter &&o) noexcept {
  if (this != &o) {
    in_sample_rate_ = o.in_sample_rate_;
    out_sample_rate_ = o.out_sample_rate_;
    sw_resample_context_ = o.sw_resample_context_;
    o.sw_resample_context_ = nullptr;
  }
}

AudioConverter::~AudioConverter() {
  if (sw_resample_context_) {
    swr_free(&sw_resample_context_);
  }
}

absl::Status AudioConverter::Convert(const AudioFrame *in_frame,
                                     AudioFrame *out_frame) {

  /* convert samples from native format to destination codec format, using the
   * resampler */
  /* compute destination number of samples */
  int dst_nb_samples =
      av_rescale_rnd(swr_get_delay(sw_resample_context_, in_sample_rate_) +
                         in_frame->c_frame()->nb_samples,
                     out_sample_rate_, in_sample_rate_, AV_ROUND_UP);

  /* when we pass a frame to the encoder, it may keep a reference to it
   * internally;
   * make sure we do not overwrite it here
   */
  if (int ret = av_frame_make_writable(out_frame->c_frame()); ret < 0) {
    return absl::AbortedError("Could not make out frame writable");
  }

  /* convert to destination format */
  if (int ret = swr_convert(sw_resample_context_, out_frame->c_frame()->data,
                            dst_nb_samples,
                            (const uint8_t **)in_frame->c_frame()->data,
                            in_frame->c_frame()->nb_samples);
      ret < 0) {
    return absl::AbortedError("Error while converting.");
  }

  out_frame->SetPTS(av_rescale_q(in_frame->GetPTS(),
                                 AVRational{1, in_sample_rate_},
                                 AVRational{1, out_sample_rate_}));

  return absl::OkStatus();
}
} // namespace media
} // namespace aikit
