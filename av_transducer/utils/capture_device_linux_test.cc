
#include "av_transducer/utils/container.h"
#include "gtest/gtest.h"

TEST(TestFFmpegUtils, CheckCaptureAudio) {
  auto res =
      aikit::media::ContainerStreamContext::CaptureDevice("pulse", "default");
  EXPECT_TRUE(res.ok());
}

TEST(TestFFmpegUtils, DISABLED_CheckCaptureScreen) {
  auto res = aikit::media::ContainerStreamContext::CaptureDevice(
      "x11grab", ":0.0+100,200");
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

// TEST(TestFFmpegUtils, DISABLED_ReadImageFrameCheck) {
//   auto res = aikit::utils::CaptureDevice("avfoundation", "3:");
//   EXPECT_TRUE(res.ok());

//   AVFrame *frame = av_frame_alloc();
//   EXPECT_TRUE(frame != nullptr);

//   AVPacket *packet = av_packet_alloc();
//   EXPECT_TRUE(packet != nullptr);

//   int n = 3;
//   while (n > 0 && av_read_frame(res->format_context, packet) == 0) {
//     --n;
//     if (packet->stream_index == res->image_stream_context->stream_index) {
//       auto s = aikit::utils::PacketToFrame(
//           res->image_stream_context->codec_context, packet, frame);

//       if (!s.ok()) {
//         EXPECT_TRUE(absl::IsFailedPrecondition(s));
//       } else {

//         auto image_data = aikit::utils::ReadImageFrame(
//             res->image_stream_context.value(), frame);
//         EXPECT_TRUE(image_data.ok()) << image_data.status().message();
//       }
//     }
//   }

//   av_packet_free(&packet);
//   av_frame_free(&frame);
//   aikit::utils::DestroyVideoStreamContext(res.value());
// }
