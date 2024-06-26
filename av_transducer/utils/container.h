#pragma once

#include "absl/status/statusor.h"
#include <cstdint>
#include <cwchar>
#include <optional>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"

namespace aikit {
namespace media {
class ContainerStreamContext {
public:
  static absl::StatusOr<ContainerStreamContext>
  CreateReaderContainerStreamContext(const std::string &url,
                                     const AVInputFormat *input_format);

  static absl::StatusOr<ContainerStreamContext>
  CreateWriterContainerStreamContext(
      AudioStreamParameters audio_stream_parameters,
      VideoStreamParameters video_stream_parameters, const std::string &url);

  ContainerStreamContext(const ContainerStreamContext &) = delete;
  ContainerStreamContext(ContainerStreamContext &&) noexcept;
  ContainerStreamContext &operator=(const ContainerStreamContext &) = delete;
  ContainerStreamContext &operator=(ContainerStreamContext &&) noexcept;

  ~ContainerStreamContext();

  AudioStreamParameters GetAudioStreamParameters();
  VideoStreamParameters GetVideoStreamParameters();

  std::unique_ptr<AudioFrame> CreateAudioFrame();
  std::unique_ptr<VideoFrame> CreateVideoFrame();

  int64_t FramePTSInMicroseconds(const AudioFrame *frame);
  void SetFramePTS(int64_t microseconds, AudioFrame *frame);
  int64_t FramePTSInMicroseconds(const VideoFrame *frame);
  void SetFramePTS(int64_t microseconds, VideoFrame *frame);

  absl::Status ReadPacket(AVPacket *packet);

  bool IsPacketAudio(AVPacket *packet);
  bool IsPacketVideo(AVPacket *packet);

  absl::Status PacketToFrame(AVPacket *packet, AudioFrame *frame);
  absl::Status PacketToFrame(AVPacket *packet, VideoFrame *frame);

  absl::Status WriteFrame(AVPacket *packet, const AudioFrame *frame);
  absl::Status WriteFrame(AVPacket *packet, const VideoFrame *frame);

  // Captures data from the device.
  // This operation is operating system dependent:
  //  MacOS: device_name is avfoundation
  //         driver_url is ":<audio_device_index>" or "<screen_device_index>:"
  //    to find audio_device_index please execute the next command
  //    ffmpeg -f avfoundation -list_devices true -i ""
  //
  //  Linux: device_name is x11grab
  //         driver_url is alsa/pulse for audio and x11grab for screen
  static absl::StatusOr<ContainerStreamContext>
  CaptureDevice(const std::string &device_name, const std::string &driver_url);

private:
  ContainerStreamContext() {};
  static absl::Status PacketToFrame(AVCodecContext *codec_context,
                                    AVPacket *packet, AVFrame *frame);

  static absl::Status WriteFrame(AVFormatContext *format_context,
                                 AVCodecContext *codec_context,
                                 int stream_index, AVPacket *packet,
                                 const AVFrame *frame);

private:
  bool is_reader_ = true;
  // Helper flag, av_write_trailer should be called only if header was written
  bool header_written_ = false;

  AVFormatContext *format_context_ = nullptr;
  std::optional<VideoStreamContext> video_stream_context_ = std::nullopt;
  std::optional<AudioStreamContext> audio_stream_context_ = std::nullopt;
};

} // namespace media
} // namespace aikit
