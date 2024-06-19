
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
#ifdef __cplusplus
}
#endif

#include "av_transducer/utils/video.h"

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

namespace aikit {
namespace media {

std::unique_ptr<VideoFrame>
VideoFrame::CreateVideoFrame(enum AVPixelFormat pix_fmt, int width,
                             int height) {
  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    return nullptr;
  }
  frame->format = pix_fmt;
  frame->width = width;
  frame->height = height;
  /* the image can be allocated by any means and av_image_alloc() is
   * just the most convenient way if av_malloc() is to be used */

  if (auto ret = av_frame_get_buffer(frame, 1); ret < 0) {
    return nullptr;
  }

  return std::unique_ptr<VideoFrame>(new VideoFrame(frame));
}

VideoFrame::VideoFrame(VideoFrame &&o) noexcept {
  av_frame_unref(c_frame_);
  av_frame_move_ref(c_frame_, o.c_frame_);
}

VideoFrame &VideoFrame::operator=(VideoFrame &&o) noexcept {
  if (this != &o) {
    av_frame_unref(c_frame_);
    av_frame_move_ref(c_frame_, o.c_frame_);
  }
}

VideoFrame::~VideoFrame() {
  if (c_frame_) {
    av_frame_free(&c_frame_);
  }
}

absl::Status VideoFrame::CopyFromBuffer(const uint8_t *buf) {
  uint8_t *src_data[4];
  int src_linesize[4];
  av_image_fill_arrays(src_data, src_linesize, buf,
                       (AVPixelFormat)c_frame_->format, c_frame_->width,
                       c_frame_->height, 1);
  av_image_copy(c_frame_->data, c_frame_->linesize, (const uint8_t **)src_data,
                src_linesize, (AVPixelFormat)c_frame_->format, c_frame_->width,
                c_frame_->height);
  return absl::OkStatus();
}

absl::Status VideoFrame::CopyToBuffer(uint8_t *buf) {
  auto dst_size = av_image_get_buffer_size(
      (AVPixelFormat)c_frame_->format, c_frame_->width, c_frame_->height, 32);
  av_image_copy_to_buffer(buf, dst_size, c_frame_->data, c_frame_->linesize,
                          (AVPixelFormat)c_frame_->format, c_frame_->width,
                          c_frame_->height, 1);

  return absl::OkStatus();
}

absl::StatusOr<VideoStreamContext> VideoStreamContext::CreateVideoStreamContext(
    const AVFormatContext *format_context, const AVCodec *codec,
    const AVCodecParameters *codec_parameters, int stream_idx) {
  VideoStreamContext result;

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

  result.format_ = (AVPixelFormat)codec_parameters->format;
  result.width_ = codec_parameters->width;
  result.height_ = codec_parameters->height;
  result.frame_rate_ = codec_parameters->framerate;

  result.codec_context_ = avcodec_alloc_context3(codec);
  if (!result.codec_context_) {
    return absl::FailedPreconditionError(
        "failed to allocated memory for AVCodecContext");
  }

  // Fill the codec context based on the values from the supplied codec
  // parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (auto res = avcodec_parameters_to_context(result.codec_context_,
                                               codec_parameters);
      res < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("failed to copy codec params to codec context. Error: ",
                     av_err2string(res)));
  }
  result.codec_context_->time_base =
      AVRational{result.frame_rate_.den, result.frame_rate_.num};
  result.codec_context_->pkt_timebase =
      AVRational{result.frame_rate_.den, result.frame_rate_.num};

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (auto res = avcodec_open2(result.codec_context_, codec, nullptr);
      res < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("failed to open codec through avcodec_open2. Error: ",
                     av_err2string(res)));
  }

  return result;
}

VideoStreamContext::VideoStreamContext(VideoStreamContext &&o) noexcept
    : stream_index_(o.stream_index_), start_time_(o.start_time_),
      time_base_(o.time_base_), frame_rate_(o.frame_rate_), width_(o.width_),
      height_(o.height_), format_(o.format_), codec_context_(o.codec_context_) {
  o.codec_context_ = nullptr;
}

VideoStreamContext &
VideoStreamContext::operator=(VideoStreamContext &&o) noexcept {
  if (this != &o) {
    stream_index_ = o.stream_index_;
    start_time_ = o.start_time_;
    time_base_ = o.time_base_;
    format_ = o.format_;
    width_ = o.width_;
    height_ = o.height_;
    frame_rate_ = o.frame_rate_;
    codec_context_ = o.codec_context_;
    o.codec_context_ = nullptr;
  }
}

VideoStreamContext::~VideoStreamContext() {
  if (codec_context_) {
    avcodec_free_context(&codec_context_);
  }
}

VideoStreamContext::VideoStreamContext() {};
} // namespace media
} // namespace aikit
