#pragma once

#include "absl/status/statusor.h"
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/frame.h"
#ifdef __cplusplus
}
#endif

namespace aikit {
namespace media {

struct AudioStreamParameters {
  int frame_size;
  int sample_rate;
  int bit_rate;
  AVSampleFormat format;
  AVChannelLayout channel_layout;

  AudioStreamParameters()
      : frame_size(1024), sample_rate(48000), bit_rate(192000),
        format(AV_SAMPLE_FMT_FLTP), channel_layout(AV_CHANNEL_LAYOUT_MONO) {}
};

class AudioFrame {
public:
  static std::unique_ptr<AudioFrame>
  CreateAudioFrame(enum AVSampleFormat sample_fmt,
                   const AVChannelLayout *channel_layout, int sample_rate,
                   int nb_samples);

  AudioFrame(const AudioFrame &) = delete;
  AudioFrame(AudioFrame &&) noexcept;
  AudioFrame &operator=(const AudioFrame &) = delete;
  AudioFrame &operator=(AudioFrame &&) noexcept;

  ~AudioFrame();

  AVFrame *c_frame() const { return c_frame_; }

  // Important: we assume here number of channels 1 and sample format FLT
  absl::Status FillAudioData(std::vector<float> &audio_data);
  // Copies frames data to the given vector.
  absl::Status AppendAudioData(std::vector<float> &audio_data) const;

  int64_t GetPTS() const { return c_frame_->pts; }
  void SetPTS(int64_t pts) { c_frame_->pts = pts; }

private:
  explicit AudioFrame(AVFrame *frame) : c_frame_(frame) {}

private:
  AVFrame *c_frame_ = nullptr;
};

class AudioStreamContext {
public:
  static absl::StatusOr<AudioStreamContext>
  CreateAudioStreamContext(const AVFormatContext *format_context,
                           const AVCodec *codec,
                           const AVCodecParameters *codec_parameters,
                           AVCodecContext *codec_context, int stream_idx);

  AudioStreamContext(const AudioStreamContext &) = delete;
  AudioStreamContext(AudioStreamContext &&) noexcept;
  AudioStreamContext &operator=(const AudioStreamContext &) = delete;
  AudioStreamContext &operator=(AudioStreamContext &&) noexcept;
  ~AudioStreamContext();

  int stream_index() { return stream_index_; }

  AVCodecContext *codec_context() { return codec_context_; }

  int sample_rate() { return sample_rate_; }
  AVChannelLayout *channel_layout() { return &channel_layout_; }
  AVSampleFormat format() { return format_; }
  AVRational time_base() { return time_base_; }

private:
  AudioStreamContext();

private:
  int stream_index_{};
  int start_time_{};
  AVRational time_base_{};
  int sample_rate_{};
  int channels_{};
  AVChannelLayout channel_layout_ = AV_CHANNEL_LAYOUT_MONO;
  AVSampleFormat format_ = AV_SAMPLE_FMT_FLT;
  AVCodecContext *codec_context_ = nullptr;
};
} // namespace media
} // namespace aikit
