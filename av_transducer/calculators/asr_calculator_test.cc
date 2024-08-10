#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "av_transducer/utils/audio.h"
#include "ml/asr/model.h"
#include "gtest/gtest.h"
#include <vector>
#include <fstream>

namespace aikit {
namespace {

class ASRCalculatorTest : public ::testing::Test {
protected:
  ASRCalculatorTest()
        : runner_(R"pb(
                      calculator: "ASRCalculator"
                      input_side_packet: "ASR_MODEL_PATH:asr_model_path"
                      input_side_packet: "SPK_MODEL_PATH:spk_model_path"
                      input_side_packet: "BUFFER_DURATION_SEC:buffer_duration_sec"
                      input_stream: "AUDIO:audio"
                      output_stream: "ASR_RESULT:asr_result"
                    )pb") {}

void SetInput() {
  runner_.MutableSidePackets()->Tag("ASR_MODEL_PATH") =
      mediapipe::MakePacket<std::string>("ml/asr/models/vosk-model-ru-0.22");
  runner_.MutableSidePackets()->Tag("SPK_MODEL_PATH") =
      mediapipe::MakePacket<std::string>("ml/asr/models/vosk-model-spk-0.4");
  runner_.MutableSidePackets()->Tag("BUFFER_DURATION_SEC") =
      mediapipe::MakePacket<int>(1);

  std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary | std::ios::ate);
  std::streamsize size = wavin.tellg();
  wavin.seekg(44, std::ios::beg);

  std::vector<float> buffer(size / sizeof(float));
  wavin.read(reinterpret_cast<char*>(buffer.data()), size);

  AVChannelLayout in_channel_layout = AV_CHANNEL_LAYOUT_MONO;
  const size_t chunk_size = 1000;

  for (size_t i = 0; i < buffer.size(); i += chunk_size) {
    size_t current_chunk_size = std::min(chunk_size, buffer.size() - i);
    std::vector<float> audio_chunk(buffer.begin() + i, buffer.begin() + i + current_chunk_size);

    auto in_frame_or = media::AudioFrame::CreateAudioFrame(
      AV_SAMPLE_FMT_FLTP, &in_channel_layout, 16000, chunk_size);
    auto status = in_frame_or->FillAudioData(audio_chunk);
    
    runner_.MutableInputs()->Tag("AUDIO").packets.push_back(
        mediapipe::Adopt(in_frame_or.release()).At(mediapipe::Timestamp(i)));
  }
}

std::vector<aikit::ml::ASRResult> GetAllOutputs() {
    std::vector<aikit::ml::ASRResult> all_results;
    for (const auto& packet : runner_.Outputs().Tag("ASR_RESULT").packets) {
      all_results.push_back(packet.Get<aikit::ml::ASRResult>());
    }
    return all_results;
  }

  mediapipe::CalculatorRunner runner_;
};

TEST_F(ASRCalculatorTest, ProcessesAudioCorrectly) {
  SetInput();
  MP_ASSERT_OK(runner_.Run());
  auto results = GetAllOutputs();

  EXPECT_GT(results.size(), 0);
  for (const auto &result : results) {
    EXPECT_FALSE(result.text.empty());
    ABSL_LOG(INFO) << "Text: " << result.text << ", Vector size: " << result.spk_embedding.size() << "\n";
  }
}

} // namespace
} // namespace aikit