#pragma once

#include "absl/status/statusor.h"
#include <memory>
#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
#ifdef __cplusplus
}
#endif

namespace aikit {
namespace media {
struct VideoStreamParameters {
  int width;
  int height;
  AVRational frame_rate;
  AVPixelFormat format;

  VideoStreamParameters()
      : width(1280), height(720), frame_rate(AVRational{25, 1}),
        format(AV_PIX_FMT_YUV420P) {}
};

class VideoFrame {
public:
  static std::unique_ptr<VideoFrame>
  CreateVideoFrame(enum AVPixelFormat pix_fmt, int width, int height);
  VideoFrame(const VideoFrame &) = delete;
  VideoFrame(VideoFrame &&) noexcept;
  VideoFrame &operator=(const VideoFrame &) = delete;
  VideoFrame &operator=(VideoFrame &&) noexcept;

  ~VideoFrame();

  // !Important. Frame should be created exactly to be
  // able to hold the given buffer amount of data.
  absl::Status CopyFromBuffer(const uint8_t *buf);
  // User has to preallocate the buf
  // use av_image_get_buffer_size to compute the
  // required size
  absl::Status CopyToBuffer(uint8_t *buf);

  AVFrame *c_frame() const { return c_frame_; }
  int64_t GetPTS() const { return c_frame_->pts; }
  void SetPTS(int64_t pts) { c_frame_->pts = pts; }

private:
  explicit VideoFrame(AVFrame *frame) : c_frame_(frame) {}

private:
  AVFrame *c_frame_ = nullptr;
};

class VideoStreamContext {

public:
  static absl::StatusOr<VideoStreamContext>
  CreateVideoStreamContext(const AVFormatContext *format_context,
                           const AVCodec *codec,
                           const AVCodecParameters *codec_parameters,
                           AVCodecContext *codec_context, int stream_idx);

  VideoStreamContext(const VideoStreamContext &) = delete;
  VideoStreamContext(VideoStreamContext &&) noexcept;
  VideoStreamContext &operator=(const VideoStreamContext &) = delete;
  VideoStreamContext &operator=(VideoStreamContext &&) noexcept;

  ~VideoStreamContext();

  int stream_index() { return stream_index_; }
  AVCodecContext *codec_context() { return codec_context_; }

private:
  VideoStreamContext();

private:
  int stream_index_{};
  int start_time_{};
  AVRational time_base_{};
  AVRational frame_rate_{};
  int width_{};
  int height_{};
  AVPixelFormat format_ = AV_PIX_FMT_YUV420P;
  AVCodecContext *codec_context_ = nullptr;
};
} // namespace media
} // namespace aikit
