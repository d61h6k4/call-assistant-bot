#include "absl/log/absl_log.h"

#include "absl/strings/str_cat.h"
#include "screenreader/utils/audio.h"
#include <cstddef>
#include <cstdint>
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
namespace {

// Replacement of av_err2str, which causes
// `error: taking address of temporary array`
// https://github.com/joncampbell123/composite-video-simulator/issues/5
#ifdef av_err2str
#undef av_err2str
#include <string>
av_always_inline std::string av_err2string(int errnum) {
  char str[AV_ERROR_MAX_STRING_SIZE];
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif // av_err2str
} // namespace

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
  audio_converter.out_sample_format_ = out_audio_stream.format;
  audio_converter.out_nb_channels_ =
      out_audio_stream.channel_layout.nb_channels;

  audio_converter.audio_fifo_ = av_audio_fifo_alloc(
      out_audio_stream.format, out_audio_stream.channel_layout.nb_channels, 1);
  if (!audio_converter.audio_fifo_) {
    return absl::AbortedError("Could not allocate Audio FIFO.");
  }
  audio_converter.out_frame_size_ = out_audio_stream.frame_size;

  return audio_converter;
}

AudioConverter::AudioConverter(AudioConverter &&o) noexcept
    : last_frame_was_already_loaded_(o.last_frame_was_already_loaded_),
      in_sample_rate_(o.in_sample_rate_), out_sample_rate_(o.out_sample_rate_),
      out_nb_channels_(o.out_nb_channels_),
      out_sample_format_(o.out_sample_format_),
      sw_resample_context_(o.sw_resample_context_),
      out_frame_size_(o.out_frame_size_), audio_fifo_(o.audio_fifo_) {
  o.sw_resample_context_ = nullptr;
  o.audio_fifo_ = nullptr;
}

AudioConverter &AudioConverter::operator=(AudioConverter &&o) noexcept {
  if (this != &o) {
    in_sample_rate_ = o.in_sample_rate_;
    out_sample_rate_ = o.out_sample_rate_;
    sw_resample_context_ = o.sw_resample_context_;
    o.sw_resample_context_ = nullptr;

    audio_fifo_ = o.audio_fifo_;
    o.audio_fifo_ = nullptr;
    out_frame_size_ = o.out_frame_size_;

    out_sample_format_ = o.out_sample_format_;
    out_nb_channels_ = o.out_nb_channels_;

    last_frame_was_already_loaded_ = o.last_frame_was_already_loaded_;
  }
}

AudioConverter::~AudioConverter() {
  if (sw_resample_context_) {
    swr_free(&sw_resample_context_);
  }
  if (audio_fifo_) {
    av_audio_fifo_free(audio_fifo_);
  }
}

absl::Status AudioConverter::Store(const AudioFrame *in_frame) {
  /* Temporary storage for the converted input samples. */
  uint8_t **converted_input_samples = nullptr;
  int frame_size = in_frame->c_frame()->nb_samples;

  /* convert samples from native format to destination codec format, using the
   * resampler */
  /* compute destination number of samples */
  int dst_nb_samples = av_rescale_rnd(
      swr_get_delay(sw_resample_context_, in_sample_rate_) + frame_size,
      out_sample_rate_, in_sample_rate_, AV_ROUND_UP);
  /* Allocate as many pointers as there are audio channels.
   * Each pointer will point to the audio samples of the corresponding
   * channels (although it may be NULL for interleaved formats).
   * Allocate memory for the samples of all channels in one consecutive
   * block for convenience. */
  if (int error = av_samples_alloc_array_and_samples(
          &converted_input_samples, nullptr, out_nb_channels_, dst_nb_samples,
          out_sample_format_, 0);
      error < 0) {
    return absl::AbortedError(
        absl::StrCat("Could not allocate converted input samples (error '%s')",
                     av_err2str(error)));
  }
  /* Convert the samples using the resampler. */
  if (int error = swr_convert(
          sw_resample_context_, converted_input_samples, dst_nb_samples,
          (const uint8_t **)in_frame->c_frame()->extended_data, frame_size);
      error < 0) {
    return absl::AbortedError(absl::StrCat(
        "Could not convert input samples. Error: ", av_err2str(error)));
  }
  /* Make the FIFO as large as it needs to be to hold both,
   * the old and the new samples. */
  if (int error = av_audio_fifo_realloc(
          audio_fifo_, av_audio_fifo_size(audio_fifo_) + dst_nb_samples);
      error < 0) {
    return absl::AbortedError("Could not reallocate FIFO");
  }

  /* Store the new samples in the FIFO buffer. */
  if (av_audio_fifo_write(audio_fifo_, (void **)converted_input_samples,
                          dst_nb_samples) < dst_nb_samples) {
    return absl::AbortedError("Could not write data to FIFO");
  }

  if (converted_input_samples) {
    av_freep(&converted_input_samples[0]);
  }
  // TODO(d61h6k4) I got error here, but in example this line exists.
  // https://ffmpeg.org/doxygen/7.0/transcode__aac_8c_source.html#l00594
  // av_freep(*converted_input_samples);

  return absl::OkStatus();
}

absl::Status AudioConverter::Load(AudioFrame *out_frame) {

  if (av_audio_fifo_size(audio_fifo_) < out_frame_size_) {
    return absl::FailedPreconditionError(
        "There is not enough data to convert yet. Store more data.");
  }

  return LoadAlways(out_frame);
}

absl::Status AudioConverter::LoadLastFrame(AudioFrame *out_frame) {
  if (last_frame_was_already_loaded_) {
    return absl::AbortedError("Last frame was already loaded.");
  }
  last_frame_was_already_loaded_ = true;

  return LoadAlways(out_frame);
}

absl::Status AudioConverter::LoadAlways(AudioFrame *out_frame) {
  /* Use the maximum number of possible samples per frame.
   * If there is less than the maximum possible frame size in the FIFO
   * buffer use this number. Otherwise, use the maximum possible frame size. */
  const int frame_size =
      FFMIN(av_audio_fifo_size(audio_fifo_), out_frame_size_);
  /* Read as many samples from the FIFO buffer as required to fill the frame.
   * The samples are stored in the frame temporarily. */
  if (av_audio_fifo_read(audio_fifo_, (void **)out_frame->c_frame()->data,
                         frame_size) < frame_size) {
    return absl::AbortedError("Could not read data from FIFO");
  }

  out_frame->c_frame()->nb_samples = frame_size;
  return absl::OkStatus();
}

} // namespace media
} // namespace aikit
