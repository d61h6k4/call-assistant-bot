

#include "gtest/gtest.h"

#include <cstddef>
#include <vector>

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/container.h"
#include "av_transducer/utils/video.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/imgutils.h"
#ifdef __cplusplus
}
#endif

namespace {
std::vector<float> GenerateAudioData(size_t nb_samples) {
  std::vector<float> audio_data(nb_samples);

  /* init signal generator */
  auto t = 0.0;
  auto pi = 3.14159265358979323846;
  auto tincr = 2.0 * pi * 110.0 / 16000.0;
  /* increment frequency by 110 Hz per second */
  auto tincr2 = 2.0 * pi * 110.0 / 16000.0 / 16000.0;

  for (auto j = 0; j < nb_samples; ++j) {
    audio_data[j] = std::sin(t);
    t += tincr;
    tincr += tincr2;
  }

  return audio_data;
}
std::vector<uint8_t> GenerateImage(int width, int height, int frame_ix) {
  auto pix_fmt = AV_PIX_FMT_YUV420P;
  auto size = av_image_get_buffer_size(pix_fmt, width, height, 1);
  std::vector<uint8_t> image(size);
  std::array<int, 4> linesize{};
  auto res = av_image_fill_linesizes(linesize.data(), pix_fmt, width);
  EXPECT_GE(res, 0);
  /* prepare a dummy image */
  /* Y */
  for (auto y = 0; y < height; y++) {
    for (auto x = 0; x < width; x++) {
      image[y * linesize[0] + x] = x + y + frame_ix * 3;
    }
  }

  /* Cb and Cr */
  for (auto y = 0; y < height / 2; y++) {
    for (auto x = 0; x < width / 2; x++) {
      image[width * height + y * linesize[1] + x] = 128 + y + frame_ix * 2;
      image[width * height + width / 2 * height / 2 + y * linesize[2] + x] =
          64 + x + frame_ix * 5;
    }
  }
  return image;
}
} // namespace

TEST(TestContainerUtils, CheckCreateReaderContianer) {
  const std::string filename = "testdata/testvideo.mp4";

  auto container =
      aikit::media::ContainerStreamContext::CreateReaderContainerStreamContext(
          filename, nullptr);

  EXPECT_TRUE(container.ok()) << container.status().message();

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet) << "failed to allocate memory for AVPacket";

  for (absl::Status st = container->ReadPacket(packet); st.ok();
       st = container->ReadPacket(packet)) {

    if (container->IsPacketAudio(packet)) {

      auto audio_frame_or = container->CreateAudioFrame();
      EXPECT_TRUE(audio_frame_or);
      st = container->PacketToFrame(packet, audio_frame_or.get());
      EXPECT_TRUE(st.ok());
    } else if (container->IsPacketVideo(packet)) {

      auto video_frame_or = container->CreateVideoFrame();
      EXPECT_TRUE(video_frame_or);
      st = container->PacketToFrame(packet, video_frame_or.get());
      EXPECT_TRUE(st.ok());
    }
  }

  av_packet_free(&packet);
}

TEST(TestContainerUtils, CheckCreateWriterContianer) {
  const std::string filename = "/tmp/test_write_container_utils.mp4";

  auto audio_params = aikit::media::AudioStreamParameters();
  audio_params.sample_rate = 16000;
  auto video_params = aikit::media::VideoStreamParameters();
  video_params.frame_rate = {17, 1};
  auto container =
      aikit::media::ContainerStreamContext::CreateWriterContainerStreamContext(
          audio_params, video_params, filename);

  EXPECT_TRUE(container.ok()) << container.status().message();

  std::vector<aikit::media::AudioFrame *> audio_frames;
  std::vector<aikit::media::VideoFrame *> video_frames;

  int N = 17;
  absl::Status status;
  for (auto i = 0; i < N; ++i) {

    auto audio_frame_or = container->CreateAudioFrame();
    EXPECT_TRUE(audio_frame_or);
    audio_frame_or->c_frame()->pts = i * audio_params.frame_size;
    auto audio_data = GenerateAudioData(audio_params.frame_size);
    auto status = audio_frame_or->FillAudioData(audio_data);
    EXPECT_TRUE(status.ok()) << status.message();
    audio_frames.push_back(audio_frame_or.release());

    auto video_frame_or = container->CreateVideoFrame();
    EXPECT_TRUE(video_frame_or);
    auto image = GenerateImage(video_params.width, video_params.height, i);
    status = video_frame_or->CopyFromBuffer(image.data());
    EXPECT_TRUE(status.ok());
    video_frame_or->c_frame()->pts = i;

    video_frames.push_back(video_frame_or.release());
  }

  AVPacket *audio_packet = av_packet_alloc();
  EXPECT_TRUE(audio_packet) << "failed to allocate memory for AVPacket";

  AVPacket *video_packet = av_packet_alloc();
  EXPECT_TRUE(video_packet) << "failed to allocate memory for AVPacket";
  // EXPECT_EQ(audio_frames.size(), 170);
  for (auto i = 0; i < N; ++i) {
    status = container->WriteFrame(video_packet, video_frames[i]);
    EXPECT_TRUE(status.ok()) << status.message();
    status = container->WriteFrame(audio_packet, audio_frames[i]);
    EXPECT_TRUE(status.ok()) << status.message();
  }
  av_packet_free(&audio_packet);
  av_packet_free(&video_packet);
}
