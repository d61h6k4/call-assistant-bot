

#include "gtest/gtest.h"

#include <cstddef>
#include <numbers>
#include <vector>

#include "absl/log/absl_log.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/container.h"

TEST(TestContainerUtils, CheckCreateReaderContianer) {
  const std::string filename = "testdata/testvideo.mp4";

  auto container =
      aikit::media::ContainerStreamContext::CreateReaderContainerStreamContext(
          filename, nullptr);

  EXPECT_TRUE(container.ok()) << container.status().message();

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  auto audio_frame_or = container->CreateAudioFrame();
  EXPECT_TRUE(audio_frame_or);

  for (absl::Status st = container->ReadPacket(packet); st.ok();
       st = container->ReadPacket(packet)) {
    st = container->PacketToFrame(packet, audio_frame_or.get());
    EXPECT_TRUE(st.ok());
  }

  av_packet_free(&packet);
}

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

TEST(TestContainerUtils, CheckCreateWriterContianer) {
  const std::string filename = "/tmp/test_write_container_utils.mp4";

  auto params = aikit::media::AudioStreamParameters();
  auto container =
      aikit::media::ContainerStreamContext::CreateWriterContainerStreamContext(
          params, filename);

  EXPECT_TRUE(container.ok()) << container.status().message();

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  std::vector<aikit::media::AudioFrame *> audio_frames;

  absl::Status status;
  for (auto i = 0; i < 17 * 10; ++i) {

    auto audio_frame_or = container->CreateAudioFrame();
    EXPECT_TRUE(audio_frame_or);
    audio_frame_or->c_frame()->pts = (i + 1) * params.frame_size;
    auto audio_data = GenerateAudioData(params.frame_size);
    auto status = audio_frame_or->FillAudioData(audio_data);
    EXPECT_TRUE(status.ok()) << status.message();
    audio_frames.push_back(audio_frame_or.release());
  }

  EXPECT_EQ(audio_frames.size(), 170);
  for (auto i = 0; i < audio_frames.size(); ++i) {

    status = container->WriteFrame(packet, audio_frames[i]);
    EXPECT_TRUE(status.ok()) << status.message();
  }
  av_packet_free(&packet);
}
