#pragma once

#include "absl/status/statusor.h"
#include <optional>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

#include "screenreader/utils/audio.h"

namespace aikit {
namespace media {

struct AudioStreamParameters {
  int frame_size;
  int sample_rate;
  int bit_rate;
  AVSampleFormat format;
  AVChannelLayout channel_layout;

  AudioStreamParameters()
      : frame_size(1024), sample_rate(16000), bit_rate(64000),
        format(AV_SAMPLE_FMT_FLT), channel_layout(AV_CHANNEL_LAYOUT_MONO) {}
};

class ContainerStreamContext {
public:
  static absl::StatusOr<ContainerStreamContext>
  CreateReaderContainerStreamContext(const std::string &url,
                                     const AVInputFormat *input_format);

  static absl::StatusOr<ContainerStreamContext>
  CreateWriterContainerStreamContext(
      AudioStreamParameters audio_stream_parameters, const std::string &url);

  ContainerStreamContext(const ContainerStreamContext &) = delete;
  ContainerStreamContext(ContainerStreamContext &&) noexcept;
  ContainerStreamContext &operator=(const ContainerStreamContext &) = delete;
  ContainerStreamContext &operator=(ContainerStreamContext &&) noexcept;

  ~ContainerStreamContext();

  absl::StatusOr<AudioFrame> CreateAudioFrame();

  absl::Status ReadPacket(AVPacket *packet);
  absl::Status PacketToFrame(AVPacket *packet, AudioFrame &frame);

  absl::Status WriteFrame(AVPacket *packet, AudioFrame &frame);

private:
  ContainerStreamContext() {};
  static absl::Status PacketToFrame(AVCodecContext *codec_context,
                                    AVPacket *packet, AVFrame *frame);

  static absl::Status WriteFrame(AVFormatContext *format_context,
                                 AVCodecContext *codec_context,
                                 int stream_index, AVPacket *packet,
                                 AVFrame *frame);

private:
  bool is_reader_ = true;
  // Helper flag, av_write_trailer should be called only if header was written
  bool header_written_ = false;

  AVFormatContext *format_context_ = nullptr;
  // std::optional<ImageStreamContext> image_stream_context = std::nullopt;
  std::optional<AudioStreamContext> audio_stream_context_ = std::nullopt;
};

} // namespace media
} // namespace aikit
