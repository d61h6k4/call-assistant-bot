
#include "screenreader/utils/audio.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#ifdef __cplusplus
}
#endif

namespace aikit::media {

absl::StatusOr<AudioFrame>
AudioFrame::CreateAudioFrame(enum AVSampleFormat sample_fmt,
                             const AVChannelLayout *channel_layout,
                             int sample_rate, int nb_samples) {

  auto audio_frame = AudioFrame(av_frame_alloc());
  if (!audio_frame.c_frame_) {
    return absl::AbortedError("Error allocating an audio frame.");
  }
  audio_frame.c_frame_->format = sample_fmt;
  av_channel_layout_copy(&audio_frame.c_frame_->ch_layout, channel_layout);
  audio_frame.c_frame_->sample_rate = sample_rate;
  audio_frame.c_frame_->nb_samples = nb_samples;

  if (nb_samples) {
    if (av_frame_get_buffer(audio_frame.c_frame_, 1) < 0) {
      return absl::AbortedError("Error allocating an audio buffer.");
    }
  }

  return audio_frame;
}

AudioFrame::AudioFrame(AudioFrame &&o) noexcept : c_frame_(o.c_frame_) {
  o.c_frame_ = nullptr;
}

AudioFrame &AudioFrame::operator=(AudioFrame &&o) noexcept {
  if (this != &o) {
    c_frame_ = o.c_frame_;
    o.c_frame_ = nullptr;
  }
}

AudioFrame::~AudioFrame() {
  if (c_frame_) {
    av_frame_free(&c_frame_);
  }
}

absl::Status AudioFrame::FillAudioData(std::vector<float> &audio_data) {
  if (c_frame_->format != AV_SAMPLE_FMT_FLT &&
      c_frame_->format != AV_SAMPLE_FMT_FLTP) {
    return absl::AbortedError("Filling audio frame with float data supported "
                              "only for AV_SAMPLE_FMT_FLT format.");
  }

  c_frame_->nb_samples = audio_data.size();
  if (auto ret = avcodec_fill_audio_frame(
          c_frame_, 1, AV_SAMPLE_FMT_FLT, (uint8_t *)audio_data.data(),
          audio_data.size() * sizeof(float) / sizeof(uint8_t), 1);
      ret < 0) {
    return absl::AbortedError(
        absl::StrCat("Failed to fill audio frame with the given audio data. ",
                     av_err2str(ret)));
  }

  return absl::OkStatus();
}

absl::StatusOr<AudioStreamContext> AudioStreamContext::CreateAudioStreamContext(
    const AVFormatContext *format_context, const AVCodec *codec,
    const AVCodecParameters *codec_parameters, int stream_idx) {

  AudioStreamContext result;

  // the component that knows how to enCOde and DECode the stream
  // it's the codec (audio or video)
  // http://ffmpeg.org/doxygen/trunk/structAVCodec.html
  // const AVCodec *codec_ ;
  // this component describes the properties of a codec used by the stream i
  // https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html
  // AVCodecParameters *codec_parameters_ ;
  result.stream_index_ = stream_idx;
  result.start_time_ = format_context->streams[stream_idx]->start_time;
  result.time_base_ =
      static_cast<float>(format_context->streams[stream_idx]->time_base.num) /
      static_cast<float>(format_context->streams[stream_idx]->time_base.den);

  result.sample_rate_ = codec_parameters->sample_rate;
  result.channels_ = codec_parameters->ch_layout.nb_channels;

  av_channel_layout_copy(&result.channel_layout_, &codec_parameters->ch_layout);
  result.format_ = AVSampleFormat(codec_parameters->format);

  result.codec_context_ = avcodec_alloc_context3(codec);
  if (!result.codec_context_) {
    return absl::FailedPreconditionError(
        "failed to allocated memory for AVCodecContext");
  }

  // Fill the codec context based on the values from the supplied codec
  // parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (avcodec_parameters_to_context(result.codec_context_, codec_parameters) !=
      0) {
    return absl::FailedPreconditionError(
        "failed to copy codec params to codec context");
  }
  /* Some formats want stream headers to be separate. */
  if (format_context->oformat->flags & AVFMT_GLOBALHEADER) {
    result.codec_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (int ret = avcodec_open2(result.codec_context_, codec, nullptr); ret < 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "failed to open codec through avcodec_open2 ", av_err2str(ret)));
  }

  return result;
}

AudioStreamContext::AudioStreamContext(AudioStreamContext &&o) noexcept
    : stream_index_(o.stream_index_), start_time_(o.start_time_),
      time_base_(o.time_base_), sample_rate_(o.sample_rate_),
      channels_(o.channels_), channel_layout_(o.channel_layout_),
      format_(o.format_), codec_context_(o.codec_context_) {
  o.codec_context_ = nullptr;
}

AudioStreamContext &
AudioStreamContext::operator=(AudioStreamContext &&o) noexcept {
  if (this != &o) {
    stream_index_ = o.stream_index_;
    start_time_ = o.start_time_;
    time_base_ = o.time_base_;
    sample_rate_ = o.sample_rate_;
    channels_ = o.channels_;
    channel_layout_ = o.channel_layout_;
    format_ = o.format_;
    codec_context_ = o.codec_context_;
    o.codec_context_ = nullptr;
  }
}

AudioStreamContext::~AudioStreamContext() {
  if (codec_context_) {
    avcodec_free_context(&codec_context_);
  }
}
AudioStreamContext::AudioStreamContext() {};
} // namespace aikit::media
