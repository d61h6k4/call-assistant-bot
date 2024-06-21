

#include "absl/log/absl_log.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/imgutils.h"
#ifdef __cplusplus
}
#endif

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace aikit {

class FFMPEGSinkVideoCalculatorTest : public ::testing::Test {
protected:
  FFMPEGSinkVideoCalculatorTest()
      : runner_(R"pb(
                  calculator: "FFMPEGSinkVideoCalculator"
                  input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
                  input_side_packet: "AUDIO_HEADER:audio_header"
                  input_side_packet: "VIDEO_HEADER:video_header"
                  input_stream: "AUDIO:audio_frames"
                  input_stream: "VIDEO:video_frames"
                  )pb") {}

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

  std::vector<float> FillAudio() {
    auto nb_samples = static_cast<int>(1024);
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

  void SetInput() {
    runner_.MutableSidePackets()->Tag("OUTPUT_FILE_PATH") =
        mediapipe::MakePacket<std::string>("/tmp/testvideo.mp4");
    auto audio_header = aikit::media::AudioStreamParameters();
    runner_.MutableSidePackets()->Tag("AUDIO_HEADER") =
        mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
            audio_header);
    auto video_header = aikit::media::VideoStreamParameters();
    runner_.MutableSidePackets()->Tag("VIDEO_HEADER") =
        mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
            video_header);

    int seconds = 10;
    for (auto ix = 0;
         ix < static_cast<int>(seconds * audio_header.sample_rate / 1024.f);
         ++ix) {

      auto audio_frame = aikit::media::AudioFrame::CreateAudioFrame(
          audio_header.format, &audio_header.channel_layout,
          audio_header.sample_rate, 1024);

      auto audio_data = FillAudio();
      audio_frame->FillAudioData(audio_data);
      audio_frame->SetPTS(ix * 1024);

      auto microseconds = ix * 1000000.0 * 1024 / audio_header.sample_rate;
      runner_.MutableInputs()->Tag("AUDIO").packets.push_back(
          mediapipe::Adopt(audio_frame.release())
              .At(mediapipe::Timestamp(static_cast<int64_t>(microseconds))));
    }
    for (auto ix = 0;
         ix < static_cast<int>(seconds * video_header.frame_rate.num); ++ix) {
      auto video_frame = aikit::media::VideoFrame::CreateVideoFrame(
          video_header.format, video_header.width, video_header.height);
      auto image_data =
          GenerateImage(video_header.width, video_header.height, ix);
      video_frame->CopyFromBuffer(image_data.data());
      video_frame->SetPTS(ix);

      auto microseconds = ix * 1000000.0 / video_header.frame_rate.num;
      runner_.MutableInputs()->Tag("VIDEO").packets.push_back(
          mediapipe::Adopt(video_frame.release())
              .At(mediapipe::Timestamp(static_cast<int64_t>(microseconds))));
    }
  }

  mediapipe::CalculatorRunner runner_;
};

TEST_F(FFMPEGSinkVideoCalculatorTest, SanityCheck) {
  SetInput();
  auto res = runner_.Run();
  EXPECT_TRUE(res.ok()) << res.message();
}
} // namespace aikit
