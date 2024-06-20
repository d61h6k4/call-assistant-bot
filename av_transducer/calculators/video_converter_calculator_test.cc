

#include "absl/log/absl_log.h"
#include "av_transducer/utils/video.h"
#include "av_transducer/utils/container.h"
#include "av_transducer/utils/audio.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "gtest/gtest.h"

namespace aikit {
namespace {

class VideoConverterCalculatorTest : public ::testing::Test {
protected:
  VideoConverterCalculatorTest()
      : runner_(R"pb(
                      calculator: "VideoConverterCalculator"
                      input_side_packet: "IN_VIDEO_HEADER:in_video_header"
                      input_side_packet: "OUT_VIDEO_HEADER:out_video_header"
                      input_stream: "IN_VIDEO:in_video"
                      output_stream: "OUT_VIDEO:out_video"
                    )pb") {}

  void SetInput(const media::VideoStreamParameters &out_video_header) {

    const std::string filename = "testdata/testvideo.mp4";

    auto container = aikit::media::ContainerStreamContext::
        CreateReaderContainerStreamContext(filename, nullptr);

    EXPECT_TRUE(container.ok()) << container.status().message();

    auto in_video_stream = container->GetVideoStreamParameters();
    runner_.MutableSidePackets()->Tag("IN_VIDEO_HEADER") =
        mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
            in_video_stream);

    runner_.MutableSidePackets()->Tag("OUT_VIDEO_HEADER") =
        mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
            out_video_header);

    packet_ = av_packet_alloc();
    EXPECT_TRUE(packet_) << "failed to allocate memory for AVPacket";

    for (absl::Status st = container->ReadPacket(packet_); st.ok();
         st = container->ReadPacket(packet_)) {

      if (container->IsPacketVideo(packet_)) {
        auto video_frame_or = container->CreateVideoFrame();
        EXPECT_TRUE(video_frame_or);
        st = container->PacketToFrame(packet_, video_frame_or.get());
        EXPECT_TRUE(st.ok());

        auto timestamp = mediapipe::Timestamp(video_frame_or->GetPTS());
        runner_.MutableInputs()
            ->Tag("IN_VIDEO")
            .packets.push_back(
                mediapipe::Adopt(video_frame_or.release()).At(timestamp));
      }
    }
  }

  const std::vector<mediapipe::Packet> &GetOutputs() {
    return runner_.Outputs().Tag("OUT_VIDEO").packets;
  }

  ~VideoConverterCalculatorTest() { av_packet_free(&packet_); }

  AVPacket *packet_ = nullptr;
  mediapipe::CalculatorRunner runner_;
};

TEST_F(VideoConverterCalculatorTest, VideoSanityCheck) {

  aikit::media::AudioStreamParameters out_audio_stream;
  aikit::media::VideoStreamParameters out_video_stream;

  SetInput(out_video_stream);

  MP_ASSERT_OK(runner_.Run());

  AVPacket *write_packet = av_packet_alloc();
  EXPECT_TRUE(write_packet) << "failed to allocate memory for AVPacket";

  auto write_container =
      aikit::media::ContainerStreamContext::CreateWriterContainerStreamContext(
          out_audio_stream, out_video_stream, "/tmp/test_videosanitycheck.mp4");
  EXPECT_TRUE(write_container.ok()) << write_container.status().message();

  for (auto &packet : GetOutputs()) {
    auto &video_frame = packet.Get<media::VideoFrame>();

    auto st = write_container->WriteFrame(write_packet, &video_frame);

    EXPECT_TRUE(st.ok()) << st.message();
  }

  av_packet_free(&write_packet);
}

} // namespace
} // namespace aikit
