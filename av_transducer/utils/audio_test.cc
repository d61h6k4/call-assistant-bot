

#include "gtest/gtest.h"

#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

#include "av_transducer/utils/audio.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "libavformat/avformat.h"
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


std::vector<float> GenerateAudioData(size_t nb_samples) {
  std::vector<float> audio_data(nb_samples);

  /* init signal generator */
  auto t = 0.0;
  auto tincr = 2.0 * std::numbers::pi * 110.0 / 16000.0;
  /* increment frequency by 110 Hz per second */
  auto tincr2 = 2.0 * std::numbers::pi * 110.0 / 16000.0 / 16000.0;

  for (auto j = 0; j < nb_samples; ++j) {
    audio_data[j] = std::sin(t);
    t += tincr;
    tincr += tincr2;
  }

  return audio_data;
}

TEST(TestAudioUtils, CheckFillAudioData) {
  auto audio_data = GenerateAudioData(16000);

  AVChannelLayout in_channel_layout = AV_CHANNEL_LAYOUT_MONO;
  auto in_frame_or = aikit::media::AudioFrame::CreateAudioFrame(
      AV_SAMPLE_FMT_FLTP, &in_channel_layout, 16000, 16000);
  EXPECT_TRUE(in_frame_or) << "Failed to allocate audio frame";
  auto status = in_frame_or->FillAudioData(audio_data);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(in_frame_or->c_frame()->linesize[0], 64000);
}

TEST(TestAudioUtils, CheckAppendAudioData) {
  auto audio_data = GenerateAudioData(16000);

  AVChannelLayout in_channel_layout = AV_CHANNEL_LAYOUT_MONO;
  auto in_frame_or = aikit::media::AudioFrame::CreateAudioFrame(
      AV_SAMPLE_FMT_FLT, &in_channel_layout, 16000, 16000);
  EXPECT_TRUE(in_frame_or);
  auto status = in_frame_or->FillAudioData(audio_data);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(in_frame_or->c_frame()->linesize[0], 64000);

  std::vector<float> copied_audio_data;
  status = in_frame_or->AppendAudioData(copied_audio_data);
  EXPECT_EQ(copied_audio_data.size(), 16000) << status.message();

  for (auto i = 0; i < 16000; ++i) {
    EXPECT_FLOAT_EQ(audio_data[0], copied_audio_data[0]);
  }
}

TEST(TestAudioUtils, CheckSaveAudioData) {
  const std::string filename = "/tmp/test_audio_utils.mp4";
  AVFormatContext *format_context = nullptr;
  int res = avformat_alloc_output_context2(&format_context, nullptr, nullptr,
                                           filename.c_str());
  EXPECT_GE(res, 0) << av_err2str(res);
  EXPECT_NE(format_context->oformat->audio_codec, AV_CODEC_ID_NONE);
  // Codec defined by avformat_alloc_output_context2, based on the filename
  const AVCodec *codec =
      avcodec_find_encoder(format_context->oformat->audio_codec);
  EXPECT_TRUE(codec != nullptr);
  // When we read (decode) from file, stream is already there,
  // but when we create file, we need to create a stream.
  AVStream *stream = avformat_new_stream(format_context, codec);
  stream->id = format_context->nb_streams - 1;
  stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
  stream->codecpar->codec_id = format_context->oformat->audio_codec;
  stream->codecpar->format = AV_SAMPLE_FMT_FLTP;
  stream->codecpar->sample_rate = 16000;
  stream->codecpar->bit_rate = 64000;
  AVChannelLayout audio_channel_layout = AV_CHANNEL_LAYOUT_MONO;
  av_channel_layout_copy(&stream->codecpar->ch_layout, &audio_channel_layout);
  stream->time_base = (AVRational){1, stream->codecpar->sample_rate};
  auto audio_stream_context_or =
      aikit::media::AudioStreamContext::CreateAudioStreamContext(
          format_context, codec, stream->codecpar, stream->id);
  EXPECT_TRUE(audio_stream_context_or.ok())
      << audio_stream_context_or.status().message();

  av_dump_format(format_context, 0, filename.c_str(), 1);
  /* open the output file, if needed */
  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {

    int ret = avio_open(&format_context->pb, filename.c_str(), AVIO_FLAG_WRITE);
    EXPECT_GE(ret, 0) << "Could not open " << filename << " "
                      << av_err2str(ret);
  }

  /* Write the stream header, if any. */
  int ret = avformat_write_header(format_context, nullptr);
  EXPECT_GE(ret, 0) << "Error occurred when opening output file: "
                    << av_err2str(ret);

  AVPacket *tmp_pkt = av_packet_alloc();
  EXPECT_TRUE(tmp_pkt) << "Could not allocate AVPacket";

  auto in_frame_or = aikit::media::AudioFrame::CreateAudioFrame(
      AV_SAMPLE_FMT_FLTP, &audio_channel_layout, 16000, 16000);
  EXPECT_TRUE(in_frame_or);

  for (auto i = 0; i < 16; ++i) {
    in_frame_or->c_frame()->pts = (i + 1) * 1024;
    auto audio_data = GenerateAudioData(1024);
    auto status = in_frame_or->FillAudioData(audio_data);
    EXPECT_TRUE(status.ok()) << status.message();

    // send the frame to the encoder
    res = avcodec_send_frame(audio_stream_context_or->codec_context(),
                             in_frame_or->c_frame());
    EXPECT_GE(res, 0) << "Error sending a frame to the encoder: "
                      << av_err2str(res);

    int ret = 0;
    while (ret >= 0) {
      ret = avcodec_receive_packet(audio_stream_context_or->codec_context(),
                                   tmp_pkt);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break;
      }
      EXPECT_GE(ret, 0) << "Error encoding a frame: " << av_err2str(ret);

      /* rescale output packet timestamp values from codec to stream timebase
       */
      // av_packet_rescale_ts(tmp_pkt, c->time_base, st->time_base);
      tmp_pkt->stream_index = stream->index;

      /* Write the compressed frame to the media file. */
      // LogPacket(fmt_ctx, pkt);

      ret = av_interleaved_write_frame(format_context, tmp_pkt);
      /* pkt is now blank (av_interleaved_write_frame() takes ownership of
       * its contents and resets pkt), so that no unreferencing is necessary.
       * This would be different if one used av_write_frame(). */
      EXPECT_GE(ret, 0) << "Error while writing output packet: "
                        << av_err2str(ret);
    }

    EXPECT_NE(ret, AVERROR_EOF) << "Encoding is finished.";
  }

  av_write_trailer(format_context);

  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
    // Close the output file
    avio_closep(&format_context->pb);
  }
  avformat_free_context(format_context);
}
