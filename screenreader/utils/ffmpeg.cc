#ifdef __cplusplus
extern "C" {
#endif
#include "libavdevice/avdevice.h"
#ifdef __cplusplus
}
#endif

#include "absl/strings/str_cat.h"
#include "screenreader/utils/ffmpeg.h"

namespace aikit::utils {

// Here we sample rate 16kHz, because distilWhisper expects 16kHz sample rate
// audios.
constexpr int kDestSampleRate = 16000;

absl::StatusOr<AudioStreamContext> InitAudioStreamContext(
    const AVFormatContext *format_context, const AVCodec *codec,
    const AVCodecParameters *codec_parameters, int stream_idx) {

  AudioStreamContext result;

  // the component that knows how to enCOde and DECode the stream
  // it's the codec (audio or video)
  // http://ffmpeg.org/doxygen/trunk/structAVCodec.html
  // const AVCodec *codec_ ;
  // this component describes the properties of a codec used by the stream i
  // https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html
  // AVCodecParameters *codec_parameters_ ;
  result.stream_index = stream_idx;
  result.start_time = format_context->streams[stream_idx]->start_time;
  result.time_base =
      static_cast<float>(format_context->streams[stream_idx]->time_base.num) /
      static_cast<float>(format_context->streams[stream_idx]->time_base.den);

  result.sample_rate = codec_parameters->sample_rate;
  result.channels = codec_parameters->ch_layout.nb_channels;
  result.format = AVSampleFormat(codec_parameters->format);

  result.codec_context = avcodec_alloc_context3(codec);
  if (!result.codec_context) {
    return absl::FailedPreconditionError(
        "failed to allocated memory for AVCodecContext");
  }

  // Fill the codec context based on the values from the supplied codec
  // parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (avcodec_parameters_to_context(result.codec_context, codec_parameters) !=
      0) {
    return absl::FailedPreconditionError(
        "failed to copy codec params to codec context");
  }

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (avcodec_open2(result.codec_context, codec, nullptr) != 0) {
    return absl::FailedPreconditionError(
        "failed to open codec through avcodec_open2");
  }

  AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_MONO;
  if (swr_alloc_set_opts2(&(result.swr_context), &(out_ch_layout),
                          AV_SAMPLE_FMT_FLT, kDestSampleRate,
                          &(codec_parameters->ch_layout),
                          result.codec_context->sample_fmt,
                          result.codec_context->sample_rate, 0, nullptr) != 0) {
    return absl::FailedPreconditionError("Failed to allocate resample context");
  }

  int error = swr_init(result.swr_context);
  if (error < 0) {
    return absl::AbortedError("Failed to initialize resample context");
  }

  return result;
}

absl::StatusOr<ImageStreamContext> InitImageStreamContext(
    const AVFormatContext *format_context, const AVCodec *codec,
    const AVCodecParameters *codec_parameters, int stream_idx) {

  ImageStreamContext result;

  result.stream_index = stream_idx;
  // the component that knows how to enCOde and DECode the stream
  // it's the codec (audio or video)
  // http://ffmpeg.org/doxygen/trunk/structAVCodec.html
  // const AVCodec *codec_ ;
  // this component describes the properties of a codec used by the stream i
  // https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html
  // AVCodecParameters *codec_parameters_ ;

  result.start_time = format_context->streams[stream_idx]->start_time;
  result.time_base =
      static_cast<float>(format_context->streams[stream_idx]->time_base.num) /
      static_cast<float>(format_context->streams[stream_idx]->time_base.den);

  result.format = AVPixelFormat(codec_parameters->format);
  result.width = codec_parameters->width;
  result.height = codec_parameters->height;
  result.frame_rate = format_context->streams[stream_idx]->r_frame_rate.num;

  result.codec_context = avcodec_alloc_context3(codec);
  if (!result.codec_context) {
    return absl::FailedPreconditionError(
        "failed to allocated memory for AVCodecContext");
  }

  // Fill the codec context based on the values from the supplied codec
  // parameters
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#gac7b282f51540ca7a99416a3ba6ee0d16
  if (avcodec_parameters_to_context(result.codec_context, codec_parameters) !=
      0) {
    return absl::FailedPreconditionError(
        "failed to copy codec params to codec context");
  }

  // Initialize the AVCodecContext to use the given AVCodec.
  // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
  if (avcodec_open2(result.codec_context, codec, nullptr) != 0) {
    return absl::FailedPreconditionError(
        "failed to open codec through avcodec_open2");
  }

  return result;
}

absl::StatusOr<VideoStreamContext>
CreateVideoStreamContext(const std::string &url,
                         const AVInputFormat *input_format) {

  VideoStreamContext video_stream_context;
  video_stream_context.format_context = avformat_alloc_context();
  // Open the file and read its header. The codecs are not opened.
  // The function arguments are:
  // AVFormatContext (the component we allocated memory for),
  // url (filename),
  // AVInputFormat (if you pass nullptr it'll do the auto detect)
  // and AVDictionary (which are options to the demuxer)
  // http://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  if (int err = avformat_open_input(&video_stream_context.format_context,
                                    url.c_str(), input_format, nullptr);
      err < 0) {
    return absl::AbortedError(absl::StrCat(
        "Failed to open the audio driver url: ", url, " Error code: ", err));
  }

  // read Packets from the Format to get stream information
  // this function populates pFormatContext->streams
  // (of size equals to pFormatContext->nb_streams)
  // the arguments are:
  // the AVFormatContext
  // and options contains options for codec corresponding to i-th stream.
  // On return each dictionary will be filled with options that were not found.
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#gad42172e27cddafb81096939783b157bb
  if (avformat_find_stream_info(video_stream_context.format_context, nullptr) !=
      0) {
    return absl::AbortedError("Failed to get the stream info");
  }

  for (int i = 0; i < video_stream_context.format_context->nb_streams; ++i) {
    AVCodecParameters *local_codec_parameters =
        video_stream_context.format_context->streams[i]->codecpar;
    // finds the registered decoder for a codec ID
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
    const AVCodec *local_codec =
        avcodec_find_decoder(local_codec_parameters->codec_id);

    if (local_codec == nullptr) {
      // In this example if the codec is not found we just skip it
      continue;
    }

    if (local_codec_parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      // Here we take first stream of type video as a image stream, can we do
      // better? is first always better?
      if (!video_stream_context.image_stream_context.has_value()) {

        auto res =
            InitImageStreamContext(video_stream_context.format_context,
                                   local_codec, local_codec_parameters, i);
        if (!res.ok()) {
          auto s =
              absl::AbortedError("Failed to initialize image stream context.");
          s.Update(res.status());
          return s;
        }

        video_stream_context.image_stream_context = res.value();
      }
    } else if (local_codec_parameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      if (!video_stream_context.audio_stream_context.has_value()) {

        auto res =
            InitAudioStreamContext(video_stream_context.format_context,
                                   local_codec, local_codec_parameters, i);
        if (!res.ok()) {
          // It's ok not to have audio stream
          auto s = absl::FailedPreconditionError(
              "Failed to initialize audio stream context.");
          s.Update(res.status());
          return s;
        }
        video_stream_context.audio_stream_context = res.value();
      }
    }
  }
  if (!video_stream_context.image_stream_context.has_value() &&
      !video_stream_context.audio_stream_context.has_value()) {
    return absl::AbortedError(
        "Video does not contain neither image nor audio streams.");
  }

  return video_stream_context;
}

void DestroyVideoStreamContext(VideoStreamContext &video_stream_context) {

  avformat_close_input(&(video_stream_context.format_context));

  if (video_stream_context.image_stream_context.has_value()) {
    avcodec_free_context(
        &(video_stream_context.image_stream_context->codec_context));
  }

  if (video_stream_context.audio_stream_context.has_value()) {
    avcodec_free_context(
        &(video_stream_context.audio_stream_context->codec_context));
    swr_free(&(video_stream_context.audio_stream_context->swr_context));
  }
}

absl::StatusOr<VideoStreamContext>
CaptureDevice(const std::string &device_name, const std::string &driver_url) {
  // Without this call we get Protocol not found error.
  avdevice_register_all();

  const AVInputFormat *input_format = av_find_input_format(device_name.c_str());
  return CreateVideoStreamContext(driver_url, input_format);
}
} // namespace aikit::utils
