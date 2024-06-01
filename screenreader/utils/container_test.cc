

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

#include "screenreader/utils/audio.h"
#include "screenreader/utils/container.h"

TEST(TestContainerUtils, CheckCreateReaderContianer) {
  const std::string filename = "/tmp/test_audio_utils.mp4";

  auto container =
      aikit::media::ContainerStreamContext::CreateReaderContainerStreamContext(
          filename, nullptr);

  EXPECT_TRUE(container.ok()) << container.status().message();

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  auto audio_frame_or = container->CreateAudioFrame();
  EXPECT_TRUE(audio_frame_or.ok()) << audio_frame_or.status().message();

  for (absl::Status st = container->ReadPacket(packet); st.ok();
       st = container->ReadPacket(packet)) {
    st = container->PacketToFrame(packet, audio_frame_or.value());
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

  auto audio_frame_or = container->CreateAudioFrame();
  EXPECT_TRUE(audio_frame_or.ok()) << audio_frame_or.status().message();

  absl::Status status;
  for (auto i = 0; i < 16; ++i) {
    audio_frame_or->c_frame()->pts = (i + 1) * params.frame_size;
    auto audio_data = GenerateAudioData(params.frame_size);
    auto status = audio_frame_or->FillAudioData(audio_data);
    EXPECT_TRUE(status.ok()) << status.message();

    status = container->WriteFrame(packet, audio_frame_or.value());
    EXPECT_TRUE(status.ok()) << status.message();
  }

  av_packet_free(&packet);
}
