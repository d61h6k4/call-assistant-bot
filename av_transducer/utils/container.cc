
#include "av_transducer/utils/container.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include <optional>
#include <sys/unistd.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavdevice/avdevice.h"
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

namespace aikit {
namespace media {

absl::StatusOr<ContainerStreamContext>
ContainerStreamContext::CreateReaderContainerStreamContext(
    const std::string &url, const AVInputFormat *input_format) {
  ContainerStreamContext container_stream_context;
  container_stream_context.is_reader_ = true;

  container_stream_context.format_context_ = avformat_alloc_context();
  // Open the file and read its header. The codecs are not opened.
  // The function arguments are:
  // AVFormatContext (the component we allocated memory for),
  // url (filename),
  // AVInputFormat (if you pass nullptr it'll do the auto detect)
  // and AVDictionary (which are options to the demuxer)
  // http://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  if (int err = avformat_open_input(&container_stream_context.format_context_,
                                    url.c_str(), input_format, nullptr);
      err < 0) {
    return absl::AbortedError(absl::StrCat("Failed to open the url: ", url,
                                           " Error: ", av_err2str(err)));
  }

  // read Packets from the Format to get stream information
  // this function populates pFormatContext->streams
  // (of size equals to pFormatContext->nb_streams)
  // the arguments are:
  // the AVFormatContext
  // and options contains options for codec corresponding to i-th stream.
  // On return each dictionary will be filled with options that were not found.
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#gad42172e27cddafb81096939783b157bb
  if (avformat_find_stream_info(container_stream_context.format_context_,
                                nullptr) != 0) {
    return absl::AbortedError("Failed to get the stream info");
  }

  for (int i = 0; i < container_stream_context.format_context_->nb_streams;
       ++i) {
    AVCodecParameters *local_codec_parameters =
        container_stream_context.format_context_->streams[i]->codecpar;
    // finds the registered decoder for a codec ID
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
    const AVCodec *local_codec =
        avcodec_find_decoder(local_codec_parameters->codec_id);

    if (local_codec == nullptr) {
      // In this example if the codec is not found we just skip it
      continue;
    }

    if (local_codec_parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      if (!container_stream_context.video_stream_context_.has_value()) {

        AVCodecContext *codec_context = avcodec_alloc_context3(local_codec);
        if (!codec_context) {
          return absl::FailedPreconditionError(
              "failed to allocated memory for AVCodecContext");
        }

        // Fill the codec context based on the values from the supplied codec
        // parameters
        // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
        if (auto res = avcodec_parameters_to_context(codec_context,
                                                     local_codec_parameters);
            res < 0) {
          return absl::FailedPreconditionError(absl::StrCat(
              "failed to copy codec params to codec context. Error: ",
              av_err2string(res)));
        }

        // Initialize the AVCodecContext to use the given AVCodec.
        // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
        if (auto res = avcodec_open2(codec_context, local_codec, nullptr);
            res < 0) {
          return absl::FailedPreconditionError(absl::StrCat(
              "failed to open codec through avcodec_open2. Error: ",
              av_err2string(res)));
        }

        auto res = VideoStreamContext::CreateVideoStreamContext(
            container_stream_context.format_context_, local_codec,
            local_codec_parameters, codec_context, i);
        if (!res.ok()) {
          auto s =
              absl::AbortedError("Failed to initialize video stream context.");
          s.Update(res.status());
          return s;
        }

        container_stream_context.video_stream_context_ = std::move(res.value());
      }
      continue;
    } else if (local_codec_parameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      if (!container_stream_context.audio_stream_context_.has_value()) {

        auto res = AudioStreamContext::CreateAudioStreamContext(
            container_stream_context.format_context_, local_codec,
            local_codec_parameters, i);
        if (!res.ok()) {
          auto s = absl::FailedPreconditionError(
              "Failed to initialize audio stream context.");
          s.Update(res.status());
          return s;
        }
        container_stream_context.audio_stream_context_ = std::move(res.value());
      }
    }
  }
  if (!container_stream_context.video_stream_context_.has_value() &&
      !container_stream_context.audio_stream_context_.has_value()) {
    return absl::AbortedError(
        "Video does not contain neither video nor audio streams.");
  }

  return container_stream_context;
}

absl::StatusOr<ContainerStreamContext>
ContainerStreamContext::CreateWriterContainerStreamContext(
    AudioStreamParameters audio_stream_parameters,
    VideoStreamParameters video_stream_parameters, const std::string &url) {
  ContainerStreamContext container_stream_context;
  container_stream_context.is_reader_ = false;

  avformat_alloc_output_context2(&container_stream_context.format_context_,
                                 nullptr, nullptr, url.c_str());
  if (!container_stream_context.format_context_) {
    ABSL_LOG(WARNING)
        << "Could not deduce output format from file extension: using MPEG";
    avformat_alloc_output_context2(&container_stream_context.format_context_,
                                   nullptr, "mpeg", url.c_str());
  }
  if (!container_stream_context.format_context_) {
    return absl::AbortedError("Failed to allocate output media context");
  }

  if (container_stream_context.format_context_->oformat->audio_codec !=
      AV_CODEC_ID_NONE) {

    // Codec defined by avformat_alloc_output_context2, based on the filename
    const AVCodec *codec = avcodec_find_encoder(
        container_stream_context.format_context_->oformat->audio_codec);

    if (!codec) {
      return absl::AbortedError("Failed to find encoder for audio.");
    }

    // This call creates and connects stream with AVFormatContext
    AVStream *stream =
        avformat_new_stream(container_stream_context.format_context_, nullptr);
    stream->id = container_stream_context.format_context_->nb_streams - 1;
    // Here we define format of the output format
    stream->codecpar->frame_size = audio_stream_parameters.frame_size;
    stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    stream->codecpar->codec_id =
        container_stream_context.format_context_->oformat->audio_codec;
    stream->codecpar->format = audio_stream_parameters.format;
    stream->codecpar->sample_rate = audio_stream_parameters.sample_rate;
    stream->codecpar->bit_rate = audio_stream_parameters.bit_rate;
    AVChannelLayout audio_channel_layout =
        audio_stream_parameters.channel_layout;
    av_channel_layout_copy(&stream->codecpar->ch_layout, &audio_channel_layout);
    stream->time_base = (AVRational){1, stream->codecpar->sample_rate};

    auto audio_stream_context_or = AudioStreamContext::CreateAudioStreamContext(
        container_stream_context.format_context_, codec, stream->codecpar,
        stream->id);

    if (!audio_stream_context_or.ok()) {
      auto s = absl::AbortedError("Failed to initialize audio stream context.");
      s.Update(audio_stream_context_or.status());
      return s;
    }

    /* Some formats want stream headers to be separate. */
    if (container_stream_context.format_context_->oformat->flags &
        AVFMT_GLOBALHEADER) {
      audio_stream_context_or->codec_context()->flags |=
          AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    container_stream_context.audio_stream_context_ =
        std::move(audio_stream_context_or.value());
  }

  /* Add the audio and video streams using the default format codecs
   * and initialize the codecs. */
  if (container_stream_context.format_context_->oformat->video_codec !=
      AV_CODEC_ID_NONE) {

    // Codec defined by avformat_alloc_output_context2, based on the filename
    const AVCodec *codec = avcodec_find_encoder(
        container_stream_context.format_context_->oformat->video_codec);

    if (!codec) {
      return absl::AbortedError("Failed to find encoder for video.");
    }

    // This call creates and connects stream with AVFormatContext
    AVStream *stream =
        avformat_new_stream(container_stream_context.format_context_, nullptr);
    stream->id = container_stream_context.format_context_->nb_streams - 1;

    AVCodecContext *codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
      return absl::FailedPreconditionError(
          "failed to allocated memory for AVCodecContext");
    }
    codec_context->codec_id =
        container_stream_context.format_context_->oformat->video_codec;
    codec_context->bit_rate = video_stream_parameters.width *
                              video_stream_parameters.height *
                              video_stream_parameters.frame_rate.num * 3;
    codec_context->width = video_stream_parameters.width;
    codec_context->height = video_stream_parameters.height;

    codec_context->time_base =
        AVRational{video_stream_parameters.frame_rate.den,
                   video_stream_parameters.frame_rate.num};
    codec_context->pkt_timebase = codec_context->time_base;
    // No idea what it does. Copied from mux.c example
    codec_context->gop_size =
        12; /* emit one intra frame every twelve frames at most */
    codec_context->pix_fmt = video_stream_parameters.format;
    if (codec_context->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
      /* just for testing, we also add B-frames */
      codec_context->max_b_frames = 2;
    }
    if (codec_context->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
      /* Needed to avoid using macroblocks in which some coeffs overflow.
       * This does not happen with normal video, it just happens here as
       * the motion of the chroma plane does not match the luma plane. */
      codec_context->mb_decision = 2;
    }

    /* Some formats want stream headers to be separate. */
    if (container_stream_context.format_context_->oformat->flags &
        AVFMT_GLOBALHEADER) {
      codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Initialize the AVCodecContext to use the given AVCodec.
    // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
    if (auto res = avcodec_open2(codec_context, codec, nullptr); res < 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("failed to open codec through avcodec_open2. Error: ",
                       av_err2string(res)));
    }
    // Fill the codec context based on the values from the supplied codec
    // parameters
    // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
    if (auto res =
            avcodec_parameters_from_context(stream->codecpar, codec_context);
        res < 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "failed to copy codec params from codec context. Error: ",
          av_err2string(res)));
    }

    auto video_stream_context_or = VideoStreamContext::CreateVideoStreamContext(
        container_stream_context.format_context_, codec, stream->codecpar,
        codec_context, stream->id);

    if (!video_stream_context_or.ok()) {
      auto s = absl::AbortedError(absl::StrCat(
          "Failed to initialize video stream context. Raised from: ",
          video_stream_context_or.status().message()));
      s.Update(video_stream_context_or.status());
      return s;
    }

    container_stream_context.video_stream_context_ =
        std::move(video_stream_context_or.value());
  }

  // Print to stdout format info.
  av_dump_format(container_stream_context.format_context_, 0, url.c_str(), 1);

  /* open the output file, if needed */
  if (!(container_stream_context.format_context_->oformat->flags &
        AVFMT_NOFILE)) {
    if (int ret = avio_open(&container_stream_context.format_context_->pb,
                            url.c_str(), AVIO_FLAG_WRITE);
        ret < 0) {
      return absl::AbortedError(
          absl::StrCat("Could not open ", url, " ", av_err2str(ret)));
    }
  }

  /* Write the stream header, if any. */
  if (int ret = avformat_write_header(container_stream_context.format_context_,
                                      nullptr);
      ret < 0) {
    return absl::AbortedError(absl::StrCat(
        "Error occurred when opening output file: ", av_err2str(ret)));
  }
  container_stream_context.header_written_ = true;

  return container_stream_context;
}

ContainerStreamContext::ContainerStreamContext(
    ContainerStreamContext &&o) noexcept
    : is_reader_(o.is_reader_), header_written_(o.header_written_),
      format_context_(o.format_context_),
      video_stream_context_(std::move(o.video_stream_context_)),
      audio_stream_context_(std::move(o.audio_stream_context_)) {

  o.format_context_ = nullptr;
}

ContainerStreamContext &
ContainerStreamContext::operator=(ContainerStreamContext &&o) noexcept {
  if (this != &o) {
    is_reader_ = o.is_reader_;
    header_written_ = o.header_written_;
    audio_stream_context_ = std::move(o.audio_stream_context_);
    video_stream_context_ = std::move(o.video_stream_context_);
    format_context_ = o.format_context_;
    o.format_context_ = nullptr;
  }
}

ContainerStreamContext::~ContainerStreamContext() {
  if (format_context_) {
    if (!is_reader_) {
      if (header_written_) {
        av_write_trailer(format_context_);
      }

      if (!(format_context_->oformat->flags & AVFMT_NOFILE)) {
        // Close the output file
        avio_closep(&format_context_->pb);
      }
    }
    avformat_close_input(&format_context_);
  }
}

absl::Status
ContainerStreamContext::PacketToFrame(AVCodecContext *codec_context,
                                      AVPacket *packet, AVFrame *frame) {
  // Supply raw packet data as input to a decoder
  // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
  int response = avcodec_send_packet(codec_context, packet);
  if (response < 0) {
    return absl::AbortedError(absl::StrFormat(
        "Error while sending a packet to the decoder: %d", response));
  }
  // Return decoded output data (into a frame) from a decoder
  // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga11e6542c4e66d3028668788a1a74217c
  response = avcodec_receive_frame(codec_context, frame);
  if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Error while decoding the output data: %d", response));
  }
  return absl::OkStatus();
}

bool ContainerStreamContext::IsPacketAudio(AVPacket *packet) {
  return packet->stream_index == audio_stream_context_->stream_index();
}

bool ContainerStreamContext::IsPacketVideo(AVPacket *packet) {
  return packet->stream_index == video_stream_context_->stream_index();
}

absl::Status ContainerStreamContext::PacketToFrame(AVPacket *packet,
                                                   AudioFrame *frame) {
  return PacketToFrame(audio_stream_context_->codec_context(), packet,
                       frame->c_frame());
}

absl::Status ContainerStreamContext::PacketToFrame(AVPacket *packet,
                                                   VideoFrame *frame) {
  return PacketToFrame(video_stream_context_->codec_context(), packet,
                       frame->c_frame());
}

absl::Status ContainerStreamContext::ReadPacket(AVPacket *packet) {
  if (!is_reader_) {
    return absl::AbortedError("The container stream context was created as "
                              "writer, reading is not allowed.");
  }

  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
  int res = av_read_frame(format_context_, packet);
  for (; res == 0; res = av_read_frame(format_context_, packet)) {
    if (IsPacketAudio(packet) || IsPacketVideo(packet)) {
      return absl::OkStatus();
    }
  }

  return absl::FailedPreconditionError(
      absl::StrCat("No packet to read. Error:", av_err2str(res)));
}

absl::Status ContainerStreamContext::WriteFrame(AVFormatContext *format_context,
                                                AVCodecContext *codec_context,
                                                int stream_index,
                                                AVPacket *packet,
                                                const AVFrame *frame) {
  // send the frame to the encoder
  int ret = 0;
  if (ret = avcodec_send_frame(codec_context, frame); ret < 0) {
    return absl::AbortedError(absl::StrCat(
        "Error sending a frame to the encoder: ", av_err2str(ret)));
  }

  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_context, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      return absl::AbortedError(
          absl::StrCat("Error encoding a frame: ", av_err2str(ret)));
    }
    /* rescale output packet timestamp values from codec to stream timebase */
    av_packet_rescale_ts(packet, codec_context->time_base,
                         format_context->streams[stream_index]->time_base);
    packet->stream_index = stream_index;

    /* Write the compressed frame to the media file. */
    ret = av_interleaved_write_frame(format_context, packet);
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

absl::Status ContainerStreamContext::WriteFrame(AVPacket *packet,
                                                const AudioFrame *frame) {
  if (is_reader_) {
    return absl::AbortedError("The container stream context was created as "
                              "reader, writing is not allowed.");
  }
  return WriteFrame(format_context_, audio_stream_context_->codec_context(),
                    audio_stream_context_->stream_index(), packet,
                    frame->c_frame());
}

absl::Status ContainerStreamContext::WriteFrame(AVPacket *packet,
                                                const VideoFrame *frame) {
  if (is_reader_) {
    return absl::AbortedError("The container stream context was created as "
                              "reader, writing is not allowed.");
  }
  return WriteFrame(format_context_, video_stream_context_->codec_context(),
                    video_stream_context_->stream_index(), packet,
                    frame->c_frame());
}

absl::StatusOr<ContainerStreamContext>
ContainerStreamContext::CaptureDevice(const std::string &device_name,
                                      const std::string &driver_url) {
  // Without this call we get Protocol not found error.
  avdevice_register_all();

  const AVInputFormat *input_format = av_find_input_format(device_name.c_str());
  return CreateReaderContainerStreamContext(driver_url, input_format);
}

int64_t
ContainerStreamContext::FramePTSInMicroseconds(const AudioFrame *frame) {
  return av_rescale_q(frame->GetPTS(), audio_stream_context_->time_base(),
                      AVRational{1, 1000000});
}

void ContainerStreamContext::SetFramePTS(int64_t microseconds,
                                         AudioFrame *frame) {
  frame->SetPTS(av_rescale_q(microseconds, AVRational{1, 1000000},
                             audio_stream_context_->time_base()));
}

std::unique_ptr<AudioFrame> ContainerStreamContext::CreateAudioFrame() {
  if (!audio_stream_context_.has_value()) {
    return nullptr;
  }

  auto audio_stream_params = GetAudioStreamParameters();
  return AudioFrame::CreateAudioFrame(
      audio_stream_params.format, &audio_stream_params.channel_layout,
      audio_stream_params.sample_rate, audio_stream_params.frame_size);
}

AudioStreamParameters ContainerStreamContext::GetAudioStreamParameters() {
  auto params = AudioStreamParameters();

  if (audio_stream_context_->codec_context()->codec->capabilities &
      AV_CODEC_CAP_VARIABLE_FRAME_SIZE) {
    params.frame_size = 16000;
  } else if (audio_stream_context_->codec_context()->frame_size == 0) {
    params.frame_size = 1024;
  } else {
    params.frame_size = audio_stream_context_->codec_context()->frame_size;
  }
  params.sample_rate = audio_stream_context_->sample_rate();
  int bit_rate =
      av_get_bits_per_sample(audio_stream_context_->codec_context()->codec_id) *
      params.sample_rate;
  params.format = audio_stream_context_->format();
  av_channel_layout_copy(&params.channel_layout,
                         audio_stream_context_->channel_layout());

  return params;
}

std::unique_ptr<VideoFrame> ContainerStreamContext::CreateVideoFrame() {
  if (!video_stream_context_.has_value()) {
    return nullptr;
  }

  auto video_stream_params = GetVideoStreamParameters();
  return VideoFrame::CreateVideoFrame(video_stream_params.format,
                                      video_stream_params.width,
                                      video_stream_params.height);
}

VideoStreamParameters ContainerStreamContext::GetVideoStreamParameters() {
  auto params = VideoStreamParameters();

  params.width = video_stream_context_->codec_context()->width;
  params.height = video_stream_context_->codec_context()->height;
  params.frame_rate = video_stream_context_->codec_context()->framerate;
  params.format = video_stream_context_->codec_context()->pix_fmt;

  return params;
}
} // namespace media
} // namespace aikit
