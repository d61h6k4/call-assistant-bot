#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/subgraph.h"
#include <string_view>

#include "av_transducer/utils/audio.h"

namespace aikit {
// A VisualGraph performs extraction of data from video (images).
//
// Inputs:
//   VIDEO - media::VideoFrame
//     Image (stream of images, so video) to extract thumbnails from
// Outputs:
//   Detections - vector of detections
class AudioGraph : public mediapipe::Subgraph {
public:
  static constexpr std::string_view kInAudioHeader = "IN_AUDIO_HEADER";
  static constexpr std::string_view kInAudio = "IN_AUDIO";
  static constexpr std::string_view kOutAudioHeader = "OUT_AUDIO_HEADER";
  static constexpr std::string_view kOutTranscription = "TRANSCRIPTION";

  absl::StatusOr<mediapipe::CalculatorGraphConfig>
  GetConfig(mediapipe::SubgraphContext *sc) override {
    mediapipe::api2::builder::Graph graph;

    // Convert to 16kHz FLT
    auto &audio_converter_node = graph.AddNode("AudioConverterCalculator");
    graph.SideIn(kInAudioHeader)
        .SetName("in_audio_header")
        .Cast<aikit::media::AudioStreamParameters>() >> audio_converter_node.SideIn("IN_AUDIO_HEADER");
    
    graph.SideIn(kOutAudioHeader)
        .SetName("out_audio_header")
        .Cast<aikit::media::AudioStreamParameters>() >> audio_converter_node.SideIn("OUT_AUDIO_HEADER");

    graph.In(kInAudio) >> audio_converter_node.In("IN_AUDIO");
    auto float_16kHz_audio_stream = audio_converter_node.Out("OUT_AUDIO");

    // apply ASR
    auto &asr_node = graph.AddNode("ASRCalculator");
    float_16kHz_audio_stream >> asr_node.In("AUDIO");

    graph.SideIn("ASR_MODEL_PATH")
          .SetName("asr_model_path")
          .Cast<std::string>() >>
      asr_node.SideIn("ASR_MODEL_PATH");
    
    graph.SideIn("SPK_MODEL_PATH")
          .SetName("spk_model_path")
          .Cast<std::string>() >>
      asr_node.SideIn("SPK_MODEL_PATH");

    auto transcription = asr_node.Out("ASR_RESULT");
    transcription >> graph.Out(kOutTranscription);

    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(AudioGraph);
} // namespace aikit
