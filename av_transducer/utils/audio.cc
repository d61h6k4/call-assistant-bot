
#include "av_transducer/utils/audio.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#ifdef __cplusplus
}
#endif

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

namespace aikit::media {

std::unique_ptr<AudioFrame>
AudioFrame::CreateAudioFrame(enum AVSampleFormat sample_fmt,
                             const AVChannelLayout *channel_layout,
                             int sample_rate, int nb_samples) {

  AVFrame *c_frame = av_frame_alloc();
  if (!c_frame) {
    return nullptr;
  }

  c_frame->format = sample_fmt;
  av_channel_layout_copy(&c_frame->ch_layout, channel_layout);
  c_frame->sample_rate = sample_rate;
  c_frame->nb_samples = nb_samples;

  if (nb_samples) {
    if (av_frame_get_buffer(c_frame, 1) < 0) {
      return nullptr;
    }
  }

  return std::unique_ptr<AudioFrame>(new AudioFrame(c_frame));
}

AudioFrame::AudioFrame(AudioFrame &&o) noexcept {
  av_frame_unref(c_frame_);
  av_frame_move_ref(c_frame_, o.c_frame_);
}

AudioFrame &AudioFrame::operator=(AudioFrame &&o) noexcept {
  if (this != &o) {
    av_frame_unref(c_frame_);
    av_frame_move_ref(c_frame_, o.c_frame_);
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
    return absl::AbortedError(
        "Filling audio frame with float data supported "
        "only for AV_SAMPLE_FMT_FLT format. Please"
        "convert the frame to the AV_SAMPLE_FMT_FLT format");
  }

  if (c_frame_->ch_layout.nb_channels != 1) {
    return absl::AbortedError(
        "The existing frame expects more then 1 channel of data.");
  }

  c_frame_->nb_samples = audio_data.size();
  uint8_t *ptr = nullptr;
  ptr = reinterpret_cast<uint8_t *>(audio_data.data());
  av_samples_copy(c_frame_->extended_data, &ptr, 0, 0, audio_data.size(), 1,
                  AV_SAMPLE_FMT_FLTP);

  return absl::OkStatus();
}

absl::Status AudioFrame::AppendAudioData(std::vector<float> &audio_data) {
  if (c_frame_->format != AV_SAMPLE_FMT_FLT &&
      c_frame_->format != AV_SAMPLE_FMT_FLTP) {
    return absl::AbortedError(
        "Appending audio frame with float data supported "
        "only for AV_SAMPLE_FMT_FLT format. Please "
        "convert the frame to the AV_SAMPLE_FMT_FLT format");
  }

  int num_channels = c_frame_->ch_layout.nb_channels;
  if (num_channels != 1) {
    return absl::AbortedError(
        "Number of channels in the frame expected to be 1, but it's not.");
  }
  int bps =
      av_get_bytes_per_sample(static_cast<AVSampleFormat>(c_frame_->format));
  int plane_size = bps * c_frame_->nb_samples;

  auto append_from = audio_data.size();
  audio_data.resize(audio_data.size() + c_frame_->nb_samples);

  uint8_t *ptr = nullptr;
  ptr = reinterpret_cast<uint8_t *>(audio_data.data()) + append_from;
  std::copy_n(c_frame_->extended_data[0], plane_size, ptr);

  return absl::OkStatus();
}

absl::StatusOr<AudioStreamContext> AudioStreamContext::CreateAudioStreamContext(
    const AVFormatContext *format_context, const AVCodec *codec,
    const AVCodecParameters *codec_parameters, AVCodecContext *codec_context,
    int stream_idx) {

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
  result.time_base_ = format_context->streams[stream_idx]->time_base;

  result.sample_rate_ = codec_parameters->sample_rate;
  result.channels_ = codec_parameters->ch_layout.nb_channels;

  av_channel_layout_copy(&result.channel_layout_, &codec_parameters->ch_layout);
  result.format_ = AVSampleFormat(codec_parameters->format);
  result.codec_context_ = codec_context;

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
