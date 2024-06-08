

#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "screenreader/utils/audio.h"
#include "screenreader/utils/container.h"
#include "gtest/gtest.h"

namespace aikit {
namespace {

class AudioConverterCalculatorTest : public ::testing::Test {
protected:
  AudioConverterCalculatorTest()
      : runner_(R"pb(
                      calculator: "AudioConverterCalculator"
                      input_side_packet: "IN_AUDIO_HEADER:in_audio_header"
                      input_side_packet: "OUT_AUDIO_HEADER:out_audio_header"
                      input_stream: "IN_AUDIO:in_audio"
                      output_stream: "OUT_AUDIO:out_audio"
                    )pb") {}

  void SetInput(const media::AudioStreamParameters &out_audio_header) {

    const std::string filename = "testdata/testvideo.mp4";

    auto container = aikit::media::ContainerStreamContext::
        CreateReaderContainerStreamContext(filename, nullptr);

    EXPECT_TRUE(container.ok()) << container.status().message();

    auto in_audio_stream = container->GetAudioStreamParameters();
    runner_.MutableSidePackets()->Tag("IN_AUDIO_HEADER") =
        mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
            in_audio_stream);

    runner_.MutableSidePackets()->Tag("OUT_AUDIO_HEADER") =
        mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
            out_audio_header);

    packet_ = av_packet_alloc();
    EXPECT_TRUE(packet_) << "failed to allocate memory for AVPacket";

    for (absl::Status st = container->ReadPacket(packet_); st.ok();
         st = container->ReadPacket(packet_)) {

      auto audio_frame_or = container->CreateAudioFrame();
      EXPECT_TRUE(audio_frame_or);
      st = container->PacketToFrame(packet_, audio_frame_or.get());
      EXPECT_TRUE(st.ok());

      auto timestamp = mediapipe::Timestamp(audio_frame_or->GetPTS());
      runner_.MutableInputs()
          ->Tag("IN_AUDIO")
          .packets.push_back(
              mediapipe::Adopt(audio_frame_or.release()).At(timestamp));
    }
  }

  const std::vector<mediapipe::Packet> &GetOutputs() {
    return runner_.Outputs().Tag("OUT_AUDIO").packets;
  }

  ~AudioConverterCalculatorTest() { av_packet_free(&packet_); }

  AVPacket *packet_ = nullptr;
  mediapipe::CalculatorRunner runner_;
};

TEST_F(AudioConverterCalculatorTest, AudioSanityCheck) {

  aikit::media::AudioStreamParameters out_audio_stream;
  // out_audio_stream.sample_rate = 44100;

  SetInput(out_audio_stream);

  MP_ASSERT_OK(runner_.Run());

  AVPacket *write_packet = av_packet_alloc();
  EXPECT_TRUE(write_packet) << "failed to allocate memory for AVPacket";

  auto write_container =
      aikit::media::ContainerStreamContext::CreateWriterContainerStreamContext(
          out_audio_stream, "/tmp/testvideo.m4a");

  for (auto &packet : GetOutputs()) {
    auto &audio_frame = packet.Get<media::AudioFrame>();

    auto st = write_container->WriteFrame(write_packet, &audio_frame);

    EXPECT_TRUE(st.ok()) << st.message();
  }

  av_packet_free(&write_packet);
}

} // namespace
} // namespace aikit
