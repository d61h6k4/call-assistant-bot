
#include "screenreader/utils/container.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "screenreader/utils/audio.h"
#include <optional>

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
      // if (!container_stream_context.image_stream_context.has_value()) {

      //   auto res =
      //       InitImageStreamContext(video_stream_context.format_context,
      //                              local_codec, local_codec_parameters, i);
      //   if (!res.ok()) {
      //     auto s =
      //         absl::AbortedError("Failed to initialize image stream
      //         context.");
      //     s.Update(res.status());
      //     return s;
      //   }

      //   container_stream_context.image_stream_context = res.value();
      // }
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
  // if (!container_stream_context.image_stream_context.has_value() &&
  //     !container_stream_context.audio_stream_context.has_value()) {
  //   return absl::AbortedError(
  //       "Video does not contain neither image nor audio streams.");
  // }

  return container_stream_context;
}

absl::StatusOr<ContainerStreamContext>
ContainerStreamContext::CreateWriterContainerStreamContext(
    AudioStreamParameters audio_stream_parameters, const std::string &url) {
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

  /* Add the audio and video streams using the default format codecs
   * and initialize the codecs. */
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
        avformat_new_stream(container_stream_context.format_context_, codec);
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

    auto audio_stream_context_or =
        aikit::media::AudioStreamContext::CreateAudioStreamContext(
            container_stream_context.format_context_, codec, stream->codecpar,
            stream->id);

    if (!audio_stream_context_or.ok()) {
      auto s = absl::AbortedError("Failed to initialize audio stream context.");
      s.Update(audio_stream_context_or.status());
      return s;
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
    if (int ret = avformat_write_header(
            container_stream_context.format_context_, nullptr);
        ret < 0) {
      return absl::AbortedError(absl::StrCat(
          "Error occurred when opening output file: ", av_err2str(ret)));
    }
    container_stream_context.header_written_ = true;

    container_stream_context.audio_stream_context_ =
        std::move(audio_stream_context_or.value());
  }

  return container_stream_context;
}

ContainerStreamContext::ContainerStreamContext(
    ContainerStreamContext &&o) noexcept
    : is_reader_(o.is_reader_), format_context_(o.format_context_),
      audio_stream_context_(std::move(o.audio_stream_context_)) {
  o.format_context_ = nullptr;
}

ContainerStreamContext &
ContainerStreamContext::operator=(ContainerStreamContext &&o) noexcept {
  if (this != &o) {
    is_reader_ = o.is_reader_;
    audio_stream_context_ = std::move(o.audio_stream_context_);
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

absl::StatusOr<AudioFrame> ContainerStreamContext::CreateAudioFrame() {
  if (!audio_stream_context_.has_value()) {
    return absl::AbortedError("Failed to create audio frame. Container creates "
                              "audio frame only according to audio stream in "
                              "it, but container doesn't have audio stream.");
  }

  return AudioFrame::CreateAudioFrame(audio_stream_context_->format(),
                                      audio_stream_context_->channel_layout(),
                                      audio_stream_context_->sample_rate(), 1);
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

absl::Status ContainerStreamContext::PacketToFrame(AVPacket *packet,
                                                   AudioFrame &frame) {
  return PacketToFrame(audio_stream_context_->codec_context(), packet,
                       frame.c_frame());
}

absl::Status ContainerStreamContext::ReadPacket(AVPacket *packet) {
  if (!is_reader_) {
    return absl::AbortedError("The container stream context was created as "
                              "writer, reading is not allowed.");
  }

  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
  while (av_read_frame(format_context_, packet) == 0) {
    if (packet->stream_index == audio_stream_context_->stream_index()) {
      return absl::OkStatus();
    }
  }

  return absl::FailedPreconditionError("No packet to read.");
}

absl::Status ContainerStreamContext::WriteFrame(AVFormatContext *format_context,
                                                AVCodecContext *codec_context,
                                                int stream_index,
                                                AVPacket *packet,
                                                AVFrame *frame) {
  // send the frame to the encoder
  if (auto ret = avcodec_send_frame(codec_context, frame); ret < 0) {
    return absl::AbortedError(absl::StrCat(
        "Error sending a frame to the encoder: ", av_err2str(ret)));
  }

  int ret = 0;
  while (ret >= 0) {
    ret = avcodec_receive_packet(codec_context, packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    } else if (ret < 0) {
      return absl::AbortedError(
          absl::StrCat("Error encoding a frame: ", av_err2str(ret)));
    }
    /* rescale output packet timestamp values from codec to stream timebase */
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
                                                AudioFrame &frame) {
  if (is_reader_) {
    return absl::AbortedError("The container stream context was created as "
                              "reader, writing is not allowed.");
  }
  return WriteFrame(format_context_, audio_stream_context_->codec_context(),
                    audio_stream_context_->stream_index(), packet,
                    frame.c_frame());
}
} // namespace media
} // namespace aikit
