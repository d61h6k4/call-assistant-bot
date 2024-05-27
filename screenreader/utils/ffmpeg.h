#pragma once

#include "absl/status/statusor.h"
#include <optional>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswresample/swresample.h"
#ifdef __cplusplus
}
#endif

namespace aikit {
namespace utils {

struct YUV {
  // We use this data only with mediapipe
  // https://github.com/google/mediapipe/blob/master/mediapipe/framework/formats/yuv_image.h#L140
  // To make easy conversation from YUV to YUVImage
  // we chose to define yuv as a unique ptrs to C array
  std::unique_ptr<uint8_t[]> y = nullptr;
  std::unique_ptr<uint8_t[]> u = nullptr;
  std::unique_ptr<uint8_t[]> v = nullptr;
};

struct ImageStreamContext {
  int stream_index;
  int start_time;
  float time_base;
  int frame_rate;
  int width;
  int height;
  AVPixelFormat format;
  AVCodecContext *codec_context = nullptr;
};

struct AudioStreamContext {
  int stream_index;
  int start_time;
  float time_base;
  int sample_rate;
  int channels;
  AVSampleFormat format;
  AVCodecContext *codec_context = nullptr;
  SwrContext *swr_context = nullptr;
};

struct VideoStreamContext {
  AVFormatContext *format_context = nullptr;
  std::optional<ImageStreamContext> image_stream_context = std::nullopt;
  std::optional<AudioStreamContext> audio_stream_context = std::nullopt;
};

absl::StatusOr<VideoStreamContext>
CreateVideoStreamContext(const std::string &url,
                         const AVInputFormat *input_format);
void DestroyVideoStreamContext(VideoStreamContext &video_stream_context);

// Captures data from the device.
// This operation is operating system dependent:
//  MacOS: device_name is avfoundation
//         driver_url is ":<audio_device_index>" or "<screen_device_index>:"
//    to find audio_device_index please execute the next command
//    ffmpeg -f avfoundation -list_devices true -i ""
//
//  Linux: device_name is x11grab
//         driver_url is alsa/pulse for audio and x11grab for screen
absl::StatusOr<VideoStreamContext> CaptureDevice(const std::string &device_name,
                                                 const std::string &driver_url);

absl::Status PacketToFrame(AVCodecContext *codec_context, AVPacket *packet,
                           AVFrame *frame);

absl::StatusOr<std::vector<float>>
ReadAudioFrame(const AudioStreamContext &audio_stream_context,
               const AVFrame *frame);
absl::StatusOr<YUV>
ReadImageFrame(const ImageStreamContext &image_stream_context,
               const AVFrame *frame);

} // namespace utils
} // namespace aikit
