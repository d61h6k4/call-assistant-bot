

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
