
#include "av_transducer/utils/container.h"
#include "gtest/gtest.h"

TEST(TestFFmpegUtils, CheckCaptureAudio) {
  auto res =
      aikit::media::ContainerStreamContext::CaptureDevice("pulse", "default");
  EXPECT_TRUE(res.ok());
}

TEST(TestFFmpegUtils, CheckCaptureScreen) {
  auto res = aikit::media::ContainerStreamContext::CaptureDevice(
      "x11grab", ":1.0");
  EXPECT_TRUE(res.ok()) << res.status().message();
}

TEST(TestFFmpegUtils, ReadAudioFrameCheck) {
  auto container =
      aikit::media::ContainerStreamContext::CaptureDevice("pulse", "default");
  EXPECT_TRUE(container.ok());

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  auto audio_frame_or = container->CreateAudioFrame();
  EXPECT_TRUE(audio_frame_or);

  std::vector<float> audio_data;
  for (absl::Status st = container->ReadPacket(packet); st.ok();
       st = container->ReadPacket(packet)) {
    st = container->PacketToFrame(packet, audio_frame_or.get());
    EXPECT_TRUE(st.ok());

    audio_frame_or->AppendAudioData(audio_data);
  }

  // It captures 1 packet
  EXPECT_EQ(audio_data.size(), 512);

  av_packet_free(&packet);
}


TEST(TestFFmpegUtils, ReadImageFrameCheck) {

    auto container =
        aikit::media::ContainerStreamContext::CaptureDevice("x11grab", ":1.0");
    EXPECT_TRUE(container.ok());

    AVPacket *packet = av_packet_alloc();
    EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

    auto video_frame_or = container->CreateVideoFrame();
    EXPECT_TRUE(video_frame_or) ;

    int n = 3;
    for (absl::Status st = container->ReadPacket(packet); st.ok();
         st = container->ReadPacket(packet)) {
      st = container->PacketToFrame(packet, video_frame_or.get());
      EXPECT_TRUE(st.ok());
      if (n < 1) {
          break;
      }
      --n;
    }

    av_packet_free(&packet);
  }
