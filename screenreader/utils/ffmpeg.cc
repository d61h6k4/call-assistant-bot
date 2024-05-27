#ifdef __cplusplus
extern "C" {
#endif
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "screenreader/utils/ffmpeg.h"

absl::Status aikit::utils::CaptureDevice(const std::string &device_name,
                                         const std::string &driver_url) {
  avdevice_register_all();

  AVFormatContext *format_context = avformat_alloc_context();
  const AVInputFormat *input_format = av_find_input_format(device_name.c_str());

  // Open the file and read its header. The codecs are not opened.
  // The function arguments are:
  // AVFormatContext (the component we allocated memory for),
  // url (filename),
  // AVInputFormat (if you pass nullptr it'll do the auto detect)
  // and AVDictionary (which are options to the demuxer)
  // http://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  if (int err = avformat_open_input(&format_context, driver_url.c_str(),
                                    input_format, nullptr);
      err < 0) {
    return absl::AbortedError(
        absl::StrCat("Failed to open the audio driver url: ", driver_url,
                     " Error code: ", err));
  }

  // read Packets from the Format to get stream information
  // this function populates pFormatContext->streams
  // (of size equals to pFormatContext->nb_streams)
  // the arguments are:
  // the AVFormatContext
  // and options contains options for codec corresponding to i-th stream.
  // On return each dictionary will be filled with options that were not found.
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#gad42172e27cddafb81096939783b157bb
  if (avformat_find_stream_info(format_context, nullptr) != 0) {
    return absl::AbortedError("Failed to get the stream info");
  }

  for (int i = 0; i < format_context->nb_streams; ++i) {
    AVCodecParameters *local_codec_parameters =
        format_context->streams[i]->codecpar;
    // finds the registered decoder for a codec ID
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
    const AVCodec *local_codec =
        avcodec_find_decoder(local_codec_parameters->codec_id);

    if (local_codec == nullptr) {
      // In this example if the codec is not found we just skip it
      continue;
    }

    if (local_codec_parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      ABSL_LOG(INFO) << "Find video stream";
    } else if (local_codec_parameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      ABSL_LOG(INFO) << "Find audio stream";
    }
  }

  return absl::OkStatus();
}
