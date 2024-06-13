

#include "absl/log/absl_log.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdint>
#include <numbers>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "av_transducer/utils/audio.h"

namespace aikit {

class FFMPEGSinkVideoCalculatorTest : public ::testing::Test {
protected:
  FFMPEGSinkVideoCalculatorTest()
      : runner_(R"pb(
                  calculator: "FFMPEGSinkVideoCalculator"
                  input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
                  input_side_packet: "AUDIO_HEADER:audio_header"
                  input_stream: "AUDIO:audio_frames"
                  )pb") {}

  mediapipe::Packet FillYuvImage(int frame_index) {

    const size_t y_size = width_ * height_;
    const size_t u_size = ((width_ + 1) / 2) * ((height_ + 1) / 2);
    const size_t v_size = ((width_ + 1) / 2) * ((height_ + 1) / 2);

    auto y = std::make_unique<uint8_t[]>(y_size);
    auto u = std::make_unique<uint8_t[]>(u_size);
    auto v = std::make_unique<uint8_t[]>(v_size);

    // Y
    for (auto h = 0; h < height_; ++h) {
      for (auto w = 0; w < width_; ++w) {
        y.get()[h * width_ + w] = (w + h + frame_index * 3) % 255;
      }
    }

    // Cb and Cr
    for (auto h = 0; h < height_ / 2; ++h) {
      for (auto w = 0; w < width_ / 2; ++w) {
        u.get()[h * ((width_ + 1) / 2) + w] = (128 + h + frame_index * 2) % 255;
        v.get()[h * ((width_ + 1) / 2) + w] = (64 + w + frame_index * 5) % 255;
      }
    }

    return mediapipe::Adopt(new mediapipe::YUVImage(
        libyuv::FOURCC_I420, std::move(y), width_, std::move(u),
        (width_ + 1) / 2, std::move(v), (width_ + 1) / 2, width_, height_));
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

    // for (auto frame_ix = 0; frame_ix < 25 * 10; ++frame_ix) {
    //   auto yuv_image_packet = FillYuvImage(frame_ix);
    //   runner_.MutableInputs()
    //       ->Tag("YUV_IMAGE")
    //       .packets.push_back(
    //           yuv_image_packet.At(mediapipe::Timestamp(frame_ix)));
    // }

    for (auto ix = 0; ix < 10 * 16; ++ix) {

      auto audio_frame = aikit::media::AudioFrame::CreateAudioFrame(
          audio_header.format, &audio_header.channel_layout,
          audio_header.sample_rate, 1024);

      auto audio_data = FillAudio();
      audio_frame->FillAudioData(audio_data);
      audio_frame->SetPTS(ix * 1024);

      auto microseconds = ix * 1000000.0;
      runner_.MutableInputs()->Tag("AUDIO").packets.push_back(
          mediapipe::Adopt(audio_frame.release())
              .At(mediapipe::Timestamp(static_cast<int64_t>(microseconds))));
    }
  }

  int width_ = 1024;
  int height_ = 768;
  mediapipe::CalculatorRunner runner_;
};

TEST_F(FFMPEGSinkVideoCalculatorTest, SanityCheck) {
  SetInput();
  auto res = runner_.Run();
  EXPECT_TRUE(res.ok()) << res.message();
  // MP_ASSERT_OK(runner_.Run());
}
} // namespace aikit
