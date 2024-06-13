#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/container.h"
#include "av_transducer/utils/converter.h"

#include "absl/log/absl_log.h"

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
  EXPECT_TRUE(in_audio_frame);
  in_audio_frame->SetPTS(in_audio_stream.frame_size); // Second frame

  auto audio_data = GenerateAudioData(in_audio_stream.frame_size);
  in_audio_frame->FillAudioData(audio_data);

  aikit::media::AudioStreamParameters out_audio_stream;
  out_audio_stream.sample_rate = 44100;
  out_audio_stream.channel_layout = AV_CHANNEL_LAYOUT_STEREO;

  auto audio_converter_or = aikit::media::AudioConverter::CreateAudioConverter(
      in_audio_stream, out_audio_stream);
  EXPECT_TRUE(audio_converter_or.ok()) << audio_converter_or.status().message();

  auto s = audio_converter_or->Store(in_audio_frame.get());
  EXPECT_TRUE(s.ok()) << s.message();
  EXPECT_EQ(in_audio_frame->GetPTS(), 1024);

  // We know that 16kHz 1024 samples -> 44.1kHz 2823 samples
  // so we call load 2 times and 1 times LoadLast
  auto out_audio_frame = aikit::media::AudioFrame::CreateAudioFrame(
      out_audio_stream.format, &out_audio_stream.channel_layout,
      out_audio_stream.sample_rate, out_audio_stream.frame_size);
  EXPECT_TRUE(out_audio_frame);
  s = audio_converter_or->Load(out_audio_frame.get());
  EXPECT_TRUE(s.ok()) << s.message();
  auto out_audio_frame2 = aikit::media::AudioFrame::CreateAudioFrame(
      out_audio_stream.format, &out_audio_stream.channel_layout,
      out_audio_stream.sample_rate, out_audio_stream.frame_size);
  EXPECT_TRUE(out_audio_frame2);
  s = audio_converter_or->Load(out_audio_frame2.get());
  EXPECT_TRUE(s.ok()) << s.message();
  auto out_audio_frame3 = aikit::media::AudioFrame::CreateAudioFrame(
      out_audio_stream.format, &out_audio_stream.channel_layout,
      out_audio_stream.sample_rate, out_audio_stream.frame_size);
  EXPECT_TRUE(out_audio_frame3);
  s = audio_converter_or->LoadLastFrame(out_audio_frame3.get());
  EXPECT_TRUE(s.ok()) << s.message();
}

TEST(TestConverterUtils, CheckReadAudioConvertWrite) {

  const std::string filename = "testdata/testvideo.mp4";

  auto container =
      aikit::media::ContainerStreamContext::CreateReaderContainerStreamContext(
          filename, nullptr);

  EXPECT_TRUE(container.ok()) << container.status().message();

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  AVPacket *write_packet = av_packet_alloc();
  EXPECT_TRUE(write_packet) << "failed to allocate memory for AVPacket";

  auto in_audio_stream = container->GetAudioStreamParameters();
  auto audio_frame_or = container->CreateAudioFrame();
  EXPECT_TRUE(audio_frame_or);

  aikit::media::AudioStreamParameters out_audio_stream;
  out_audio_stream.sample_rate = 48000;
  // out_audio_stream.bit_rate = 96000;
  // out_audio_stream.frame_size =
  //     static_cast<int>(static_cast<float>(out_audio_stream.sample_rate) *
  //                      static_cast<float>(in_audio_stream.frame_size) /
  //                      static_cast<float>(in_audio_stream.sample_rate));

  auto write_container =
      aikit::media::ContainerStreamContext::CreateWriterContainerStreamContext(
          out_audio_stream, "/tmp/testvideo.m4a");
  auto out_audio_frame = write_container->CreateAudioFrame();
  EXPECT_TRUE(out_audio_frame);

  auto audio_converter_or = aikit::media::AudioConverter::CreateAudioConverter(
      in_audio_stream, out_audio_stream);
  EXPECT_TRUE(audio_converter_or.ok()) << audio_converter_or.status().message();

  for (absl::Status st = container->ReadPacket(packet); st.ok();
       st = container->ReadPacket(packet)) {
    st = container->PacketToFrame(packet, audio_frame_or.get());
    EXPECT_TRUE(st.ok());

    auto s = audio_converter_or->Store(audio_frame_or.get());
    EXPECT_TRUE(s.ok());

    s = audio_converter_or->Load(out_audio_frame.get());
    if (s.ok()) {
      st = write_container->WriteFrame(write_packet, out_audio_frame.get());
      EXPECT_TRUE(st.ok()) << st.message();
    } else {
      EXPECT_TRUE(absl::IsFailedPrecondition(s)) << s.message();
    }
  }

  av_packet_free(&packet);
  av_packet_free(&write_packet);
}
