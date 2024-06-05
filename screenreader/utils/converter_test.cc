#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <bsm/audit.h>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include "screenreader/utils/audio.h"
#include "screenreader/utils/converter.h"

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

TEST(TestConverterUtils, CheckCreateAudioConverter) {
  aikit::media::AudioStreamParameters in_audio_stream;
  aikit::media::AudioStreamParameters out_audio_stream;

  auto audio_converter_or = aikit::media::AudioConverter::CreateAudioConverter(
      in_audio_stream, out_audio_stream);

  EXPECT_TRUE(audio_converter_or.ok()) << audio_converter_or.status().message();
}

TEST(TestConverterUtils, CheckAudioConvert) {
  aikit::media::AudioStreamParameters in_audio_stream;
  auto in_audio_frame = aikit::media::AudioFrame::CreateAudioFrame(
      in_audio_stream.format, &in_audio_stream.channel_layout,
      in_audio_stream.sample_rate, in_audio_stream.frame_size);
  EXPECT_TRUE(in_audio_frame.ok());
  in_audio_frame->SetPTS(in_audio_stream.frame_size); // Second frame

  auto audio_data = GenerateAudioData(in_audio_stream.frame_size);
  in_audio_frame->FillAudioData(audio_data);

  aikit::media::AudioStreamParameters out_audio_stream;
  out_audio_stream.sample_rate = 44100;
  out_audio_stream.channel_layout = AV_CHANNEL_LAYOUT_STEREO;
  auto out_audio_frame = aikit::media::AudioFrame::CreateAudioFrame(
      out_audio_stream.format, &out_audio_stream.channel_layout,
      out_audio_stream.sample_rate, out_audio_stream.frame_size);
  EXPECT_TRUE(out_audio_frame.ok());

  auto audio_converter_or = aikit::media::AudioConverter::CreateAudioConverter(
      in_audio_stream, out_audio_stream);
  EXPECT_TRUE(audio_converter_or.ok()) << audio_converter_or.status().message();

  auto s = audio_converter_or->Convert(in_audio_frame.value(),
                                       out_audio_frame.value());
  EXPECT_EQ(in_audio_frame->GetPTS(), 1024);
  EXPECT_EQ(out_audio_frame->GetPTS(),
            static_cast<int64_t>(1024.0 / 16000.0 * 44100.0));
  EXPECT_TRUE(s.ok()) << s.message();
}
