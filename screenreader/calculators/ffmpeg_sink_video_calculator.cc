
#include "libyuv/convert.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include <cstdint>
#include <optional>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avassert.h"
#include "libavutil/channel_layout.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/timestamp.h"
#include "libswresample/swresample.h"
#include "libswscale/swscale.h"
#ifdef __cplusplus
}
#endif

namespace aikit {
namespace {
constexpr int kStreamFrameRate = 25;
constexpr int kWidth = 1024;
constexpr int kHeight = 768;

// a wrapper around a single output AVStream
struct OutputStream {
  AVStream *st;
  AVCodecContext *enc;

  /* pts of the next frame that will be generated */
  int64_t next_pts;
  int samples_count;

  AVFrame *frame;
  AVFrame *tmp_frame;

  AVPacket *tmp_pkt;

  float t, tincr, tincr2;

  struct SwsContext *sws_ctx;
  struct SwrContext *swr_ctx;
};

/* Add an output stream. */
absl::StatusOr<OutputStream>
AddStream(AVFormatContext *oc, const AVCodec **codec, enum AVCodecID codec_id) {
  OutputStream ost{};

  AVCodecContext *c = nullptr;
  int i = 0;

  /* find the encoder */
  *codec = avcodec_find_encoder(codec_id);
  if (!(*codec)) {
    return absl::AbortedError(absl::StrCat("Could not find encoder for ",
                                           avcodec_get_name(codec_id)));
  }

  ost.tmp_pkt = av_packet_alloc();
  if (!ost.tmp_pkt) {
    return absl::AbortedError("Could not allocate AVPacket");
  }

  ost.st = avformat_new_stream(oc, NULL);
  if (!ost.st) {
    return absl::AbortedError("Could not allocate stream");
  }
  ost.st->id = oc->nb_streams - 1;
  c = avcodec_alloc_context3(*codec);
  if (!c) {
    return absl::AbortedError("Could not alloc an encoding context");
  }
  ost.enc = c;

  AVChannelLayout audio_channel_layout = AV_CHANNEL_LAYOUT_STEREO;
  switch ((*codec)->type) {
  case AVMEDIA_TYPE_AUDIO:
    c->sample_fmt =
        (*codec)->sample_fmts ? (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
    c->bit_rate = 64000;
    c->sample_rate = 44100;
    if ((*codec)->supported_samplerates) {
      c->sample_rate = (*codec)->supported_samplerates[0];
      for (i = 0; (*codec)->supported_samplerates[i]; i++) {
        if ((*codec)->supported_samplerates[i] == 44100)
          c->sample_rate = 44100;
      }
    }
    av_channel_layout_copy(&c->ch_layout, &audio_channel_layout);
    ost.st->time_base = (AVRational){1, c->sample_rate};
    break;

  case AVMEDIA_TYPE_VIDEO:
    c->codec_id = codec_id;

    c->bit_rate = 400000;
    /* Resolution must be a multiple of two. */
    c->width = kWidth;
    c->height = kHeight;
    /* timebase: This is the fundamental unit of time (in seconds) in terms
     * of which frame timestamps are represented. For fixed-fps content,
     * timebase should be 1/framerate and timestamp increments should be
     * identical to 1. */
    ost.st->time_base = (AVRational){1, kStreamFrameRate};
    c->time_base = ost.st->time_base;

    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
      /* just for testing, we also add B-frames */
      c->max_b_frames = 2;
    }
    if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
      /* Needed to avoid using macroblocks in which some coeffs overflow.
       * This does not happen with normal video, it just happens here as
       * the motion of the chroma plane does not match the luma plane. */
      c->mb_decision = 2;
    }
    break;

  default:
    break;
  }

  /* Some formats want stream headers to be separate. */
  if (oc->oformat->flags & AVFMT_GLOBALHEADER) {
    c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  return ost;
}

absl::Status CloseStream(AVFormatContext *oc, OutputStream &ost) {
  avcodec_free_context(&ost.enc);
  av_frame_free(&ost.frame);
  av_frame_free(&ost.tmp_frame);
  av_packet_free(&ost.tmp_pkt);
  sws_freeContext(ost.sws_ctx);
  swr_free(&ost.swr_ctx);
}

AVFrame *AllocFrame(enum AVPixelFormat pix_fmt, int width, int height) {
  AVFrame *frame = nullptr;
  int ret = 0;

  frame = av_frame_alloc();
  if (!frame) {
    return nullptr;
  }

  frame->format = pix_fmt;
  frame->width = width;
  frame->height = height;

  /* allocate the buffers for the frame data */
  ret = av_frame_get_buffer(frame, 0);
  if (ret < 0) {
    return nullptr;
  }

  return frame;
}

absl::Status OpenVideo(AVFormatContext *oc, const AVCodec *codec,
                       OutputStream &ost) {
  int ret = 0;
  AVCodecContext *c = ost.enc;

  /* open the codec */
  ret = avcodec_open2(c, codec, nullptr);
  if (ret < 0) {
    return absl::AbortedError(
        absl::StrCat("Could not open video codec: ", av_err2str(ret)));
  }

  /* allocate and init a re-usable frame */
  ost.frame = AllocFrame(c->pix_fmt, c->width, c->height);
  if (!ost.frame) {
    return absl::AbortedError("Could not allocate video frame");
  }

  /* If the output format is not YUV420P, then a temporary YUV420P
   * picture is needed too. It is then converted to the required
   * output format. */
  ost.tmp_frame = NULL;
  if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
    ost.tmp_frame = AllocFrame(AV_PIX_FMT_YUV420P, c->width, c->height);
    if (!ost.tmp_frame) {
      absl::AbortedError("Could not allocate temporary video frame");
    }
  }

  /* copy the stream parameters to the muxer */
  ret = avcodec_parameters_from_context(ost.st->codecpar, c);
  if (ret < 0) {
    return absl::AbortedError("Could not copy the stream parameters");
  }

  return absl::OkStatus();
}

AVFrame *AllocAudioFrame(enum AVSampleFormat sample_fmt,
                         const AVChannelLayout *channel_layout, int sample_rate,
                         int nb_samples) {
  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    return nullptr;
  }

  frame->format = sample_fmt;
  av_channel_layout_copy(&frame->ch_layout, channel_layout);
  frame->sample_rate = sample_rate;
  frame->nb_samples = nb_samples;

  if (nb_samples) {
    if (av_frame_get_buffer(frame, 0) < 0) {
      return nullptr;
    }
  }

  return frame;
}

absl::Status OpenAudio(AVFormatContext *oc, const AVCodec *codec,
                       OutputStream &ost) {

  AVCodecContext *c = ost.enc;
  int nb_samples = 0;
  int ret = 0;

  /* open it */
  if (auto ret = avcodec_open2(c, codec, nullptr); ret < 0) {
    return absl::AbortedError(
        absl::StrCat("Could not open audio codec: ", av_err2str(ret)));
  }

  if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) {
    nb_samples = 10000;
  } else {
    nb_samples = c->frame_size;
  }

  ost.frame =
      AllocAudioFrame(c->sample_fmt, &c->ch_layout, c->sample_rate, nb_samples);
  if (!ost.frame) {
    return absl::AbortedError("Could no allocate audio frame.");
  }
  ost.tmp_frame = AllocAudioFrame(AV_SAMPLE_FMT_S16, &c->ch_layout,
                                  c->sample_rate, nb_samples);
  if (!ost.tmp_frame) {
    return absl::AbortedError("Could no allocate audio frame.");
  }
  /* copy the stream parameters to the muxer */
  if (auto ret = avcodec_parameters_from_context(ost.st->codecpar, c);
      ret < 0) {
    return absl::AbortedError("Could not copy the stream parameters");
  }

  /* create resampler context */
  ost.swr_ctx = swr_alloc();
  if (!ost.swr_ctx) {
    return absl::AbortedError("Could not allocate resampler context");
  }

  /* set options */
  av_opt_set_chlayout(ost.swr_ctx, "in_chlayout", &c->ch_layout, 0);
  av_opt_set_int(ost.swr_ctx, "in_sample_rate", c->sample_rate, 0);
  av_opt_set_sample_fmt(ost.swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_S16, 0);
  av_opt_set_chlayout(ost.swr_ctx, "out_chlayout", &c->ch_layout, 0);
  av_opt_set_int(ost.swr_ctx, "out_sample_rate", c->sample_rate, 0);
  av_opt_set_sample_fmt(ost.swr_ctx, "out_sample_fmt", c->sample_fmt, 0);

  /* initialize the resampling context */
  if (swr_init(ost.swr_ctx) < 0) {
    return absl::AbortedError("Failed to initialize the resampling context");
  }

  return absl::OkStatus();
}

void LogPacket(const AVFormatContext *fmt_ctx, const AVPacket *pkt) {
  AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;

  ABSL_LOG_FIRST_N(INFO, 10)
      << "pts:" << av_ts2str(pkt->pts)
      << " pts_time:" << av_ts2timestr(pkt->pts, time_base)
      << " dts:" << av_ts2str(pkt->dts)
      << " dts_time:" << av_ts2timestr(pkt->dts, time_base)
      << " duration:" << av_ts2str(pkt->duration)
      << " duration_time:" << av_ts2timestr(pkt->duration, time_base)
      << " stream_index:" << pkt->stream_index;
}

absl::Status WriteFrame(AVFormatContext *fmt_ctx, AVCodecContext *c,
                        AVStream *st, AVFrame *frame, AVPacket *pkt) {
  // send the frame to the encoder
  if (auto ret = avcodec_send_frame(c, frame); ret < 0) {
    return absl::AbortedError(absl::StrCat(
        "Error sending a frame to the encoder: ", av_err2str(ret)));
  }

  int ret = 0;
  while (ret >= 0) {
    ret = avcodec_receive_packet(c, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      return absl::AbortedError(
          absl::StrCat("Error encoding a frame: ", av_err2str(ret)));
    }

    /* rescale output packet timestamp values from codec to stream timebase */
    av_packet_rescale_ts(pkt, c->time_base, st->time_base);
    pkt->stream_index = st->index;

    /* Write the compressed frame to the media file. */
    LogPacket(fmt_ctx, pkt);
    ret = av_interleaved_write_frame(fmt_ctx, pkt);
    /* pkt is now blank (av_interleaved_write_frame() takes ownership of
     * its contents and resets pkt), so that no unreferencing is necessary.
     * This would be different if one used av_write_frame(). */
    if (ret < 0) {
      absl::AbortedError(
          absl::StrCat("Error while writing output packet: ", av_err2str(ret)));
    }
  }

  if (ret == AVERROR_EOF) {
    return absl::FailedPreconditionError("Encoding is finished.");
  }
  return absl::OkStatus();
}
} // namespace

// Calculator takes video (images) stream (optional) and audio stream
// (optional), muxes them and writes to a file.
//
// Example config:
// node {
//   calculator: "FFMPEGSinkVideoCalculator"
//   input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
//   input_stream: "YUV_IMAGE:video_frames"
//   input_stream: "AUDIO:audio_frames"
// }
class FFMPEGSinkVideoCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::SideInput<std::string> kInFilePath{
      "OUTPUT_FILE_PATH"};
  static constexpr mediapipe::api2::Input<mediapipe::YUVImage>::Optional
      kInVideo{"YUV_IMAGE"};
  static constexpr mediapipe::api2::Input<std::vector<float>>::Optional
      kInAudio{"AUDIO"};

  MEDIAPIPE_NODE_CONTRACT(kInFilePath, kInVideo, kInAudio);

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  AVFormatContext *output_media_context_ = nullptr;
  const AVOutputFormat *output_format_ = nullptr;

  std::optional<OutputStream> video_stream_ = std::nullopt;
  std::optional<OutputStream> audio_stream_ = std::nullopt;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGSinkVideoCalculator);

absl::Status FFMPEGSinkVideoCalculator::Open(mediapipe::CalculatorContext *cc) {
  const auto &output_file_path = kInFilePath(cc).Get();

  avformat_alloc_output_context2(&output_media_context_, nullptr, nullptr,
                                 output_file_path.c_str());
  if (!output_media_context_) {
    ABSL_LOG(INFO)
        << "Could not deduce output format from file extension: using MPEG";
    avformat_alloc_output_context2(&output_media_context_, nullptr, "mpeg",
                                   output_file_path.c_str());
  }
  if (!output_media_context_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Failed to allocate output media context";
  }

  output_format_ = output_media_context_->oformat;

  /* Add the audio and video streams using the default format codecs
   * and initialize the codecs. */
  const AVCodec *video_codec = nullptr;
  if (output_format_->video_codec != AV_CODEC_ID_NONE) {
    if (!kInVideo(cc).IsConnected()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Could not find video stream. Based on the ouput file path "
                "extension FFMpeg assumes user wants to store video stream, "
                "but there is no video stream connected to the node. Please "
                "rename output file path (e.g. "
                "use .aac ext) or provide video stream to the node.";
    }
    auto video_stream_or = AddStream(output_media_context_, &video_codec,
                                     output_format_->video_codec);
    if (!video_stream_or.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << video_stream_or.status().message();
    }
    video_stream_ = video_stream_or.value();
  }

  const AVCodec *audio_codec = nullptr;
  if (output_format_->audio_codec != AV_CODEC_ID_NONE) {
    if (!kInAudio(cc).IsConnected()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Could not find audio stream. Based on the ouput file path "
                "extension FFMpeg assumes user wants to store audio stream, "
                "but there is no audio stream connected to the node. Please "
                "rename output file path (e.g. "
                "use .h264 ext) or provide audio stream to the node.";
    }
    auto audio_stream_or = AddStream(output_media_context_, &audio_codec,
                                     output_format_->audio_codec);
    if (!audio_stream_or.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << audio_stream_or.status().message();
    }
    audio_stream_ = audio_stream_or.value();
  }

  /* Now that all the parameters are set, we can open the audio and
   * video codecs and allocate the necessary encode buffers. */
  if (video_stream_.has_value()) {
    auto res =
        OpenVideo(output_media_context_, video_codec, video_stream_.value());
    if (!res.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << res.message();
    }
  }
  if (audio_stream_.has_value()) {
    auto res =
        OpenAudio(output_media_context_, audio_codec, audio_stream_.value());
    if (!res.ok()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << res.message();
    }
  }

  av_dump_format(output_media_context_, 0, output_file_path.c_str(), 1);

  /* open the output file, if needed */
  if (!(output_format_->flags & AVFMT_NOFILE)) {

    if (int ret = avio_open(&output_media_context_->pb,
                            output_file_path.c_str(), AVIO_FLAG_WRITE);
        ret < 0) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Could not open " << output_file_path << " " << av_err2str(ret);
    }
  }

  /* Write the stream header, if any. */
  if (int ret = avformat_write_header(output_media_context_, nullptr);
      ret < 0) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Error occurred when opening output file: " << av_err2str(ret);
  }

  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Close(mediapipe::CalculatorContext *cc) {

  av_write_trailer(output_media_context_);
  if (video_stream_.has_value()) {
    CloseStream(output_media_context_, video_stream_.value());
  }

  if (audio_stream_.has_value()) {
    CloseStream(output_media_context_, audio_stream_.value());
  }

  if (!(output_format_->flags & AVFMT_NOFILE)) {
    // Close the output file
    avio_closep(&output_media_context_->pb);
  }
  avformat_free_context(output_media_context_);
  return absl::OkStatus();
}

absl::Status
FFMPEGSinkVideoCalculator::Process(mediapipe::CalculatorContext *cc) {

  if (kInVideo(cc).IsConnected() && !kInVideo(cc).IsEmpty()) {
    if (video_stream_.has_value()) {

      const auto &yuv_image = kInVideo(cc).Get();
      libyuv::I420Copy(
          yuv_image.data(0), yuv_image.stride(0), yuv_image.data(1),
          yuv_image.stride(1), yuv_image.data(2), yuv_image.stride(2),
          video_stream_->frame->data[0], video_stream_->frame->linesize[0],
          video_stream_->frame->data[1], video_stream_->frame->linesize[1],
          video_stream_->frame->data[2], video_stream_->frame->linesize[2],
          video_stream_->enc->width, video_stream_->enc->height);
      video_stream_->frame->pts = cc->InputTimestamp().Microseconds();

      auto res = WriteFrame(output_media_context_, video_stream_->enc,
                            video_stream_->st, video_stream_->frame,
                            video_stream_->tmp_pkt);
      if (!res.ok()) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << res.message();
      }
    } else {
      ABSL_LOG(WARNING)
          << "Could not find video stream. Based on the ouput file path "
             "extension FFMpeg assumes user wants to store only audio stream, "
             "Please rename output file path (e.g. use .mp4 ext) if you want "
             "to save video stream too.";
    }
  }

  if (kInAudio(cc).IsConnected() && !kInAudio(cc).IsEmpty()) {
    if (audio_stream_.has_value()) {

      const auto &audio_data = kInAudio(cc).Get();
      //  convert samples from native format to destination codec format,
      //  using the resampler compute destination number of samples
      auto dst_nb_samples =
          av_rescale_rnd(swr_get_delay(audio_stream_->swr_ctx,
                                       audio_stream_->enc->sample_rate) +
                             audio_data.size(),
                         audio_stream_->enc->sample_rate, 16000, AV_ROUND_UP);

      // when we pass a frame to the encoder, it may keep a reference to it
      // internally; make sure we do not overwrite it here
      if (av_frame_make_writable(audio_stream_->frame) < 0) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "Could not make audio frame writeable";
      }
      if (swr_convert(audio_stream_->swr_ctx, audio_stream_->frame->data,
                      dst_nb_samples, (const uint8_t **)(audio_data.data()),
                      audio_data.size())) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "Error while converting audio frame";
      }
      ABSL_LOG(INFO) << "Data converted";
      audio_stream_->frame->pts =
          av_rescale_q(cc->InputTimestamp().Microseconds(),
                       (AVRational){1, audio_stream_->enc->sample_rate},
                       audio_stream_->enc->time_base);
      audio_stream_->samples_count += dst_nb_samples;

      auto res = WriteFrame(output_media_context_, audio_stream_->enc,
                            audio_stream_->st, audio_stream_->frame,
                            audio_stream_->tmp_pkt);
      if (!res.ok()) {
        return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << res.message();
      }
    } else {
      ABSL_LOG(WARNING)
          << "Could not find audio stream. Based on the ouput file path "
             "extension FFMpeg assumes user wants to store only video "
             "stream, "
             "Please rename output file path (e.g. use .mp4 ext) if you want "
             "to save audio stream too.";
    }
  }

  return absl::OkStatus();
}

} // namespace aikit
