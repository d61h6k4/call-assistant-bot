
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <mutex>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/absl_log.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_graph.h"

namespace {
volatile std::sig_atomic_t SIGNAL_STATUS;
std::mutex SIGNAL_STATUS_MUTEX;
std::condition_variable SIGNAL_STATUS_COND_V;
} // namespace

void SignalHandler(int signal) {
  std::unique_lock<std::mutex> lock(SIGNAL_STATUS_MUTEX);
  SIGNAL_STATUS = signal;
  SIGNAL_STATUS_COND_V.notify_one();
}

// Absolute path to the model file when building docker image
ABSL_FLAG(
    std::string, cdetr_model_path,
    "/meeting_bot/meeting_bot.runfiles/_main/ml/detection/models/model.onnx",
    "Specify path to the CDETR model.");
ABSL_FLAG(
    std::string, ocr_model_path,
    "/meeting_bot/meeting_bot.runfiles/_main/ml/ocr/models/model.onnx",
    "Specify path to the OCR model.");

ABSL_FLAG(
    std::string, asr_model_path,
    "/meeting_bot/meeting_bot.runfiles/_main/ml/asr/models/vosk-model-ru-0.42",
    "Specify path to the ASR model.");
ABSL_FLAG(
    std::string, spk_model_path,
    "/meeting_bot/meeting_bot.runfiles/_main/ml/asr/models/vosk-model-spk-0.4",
    "Specify path to the SPK model.");

ABSL_FLAG(std::string, output_file_path, "", "Full path of video to save.");

mediapipe::CalculatorGraphConfig BuildGraph() {
  mediapipe::api2::builder::Graph graph;

  // Capture video device
  auto &capture_video_node = graph.AddNode("FFMPEGCaptureScreenCalculator");
  auto video_header = capture_video_node.SideOut("VIDEO_HEADER");
  auto video_stream = capture_video_node.Out("VIDEO");

  // Convert to YUV420P
  auto &video_converter_node = graph.AddNode("VideoConverterCalculator");
  video_header >> video_converter_node.SideIn("IN_VIDEO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      video_converter_node.SideIn("OUT_VIDEO_HEADER");
  video_stream >> video_converter_node.In("IN_VIDEO");
  auto yuv_video_stream = video_converter_node.Out("OUT_VIDEO");

  // Capture audio device
  auto &capture_audio_node = graph.AddNode("FFMPEGCaptureAudioCalculator");
  auto audio_header = capture_audio_node.SideOut("AUDIO_HEADER");
  auto audio_stream = capture_audio_node.Out("AUDIO");

  // Convert to 48kHz FLT
  auto &audio_converter_node = graph.AddNode("AudioConverterCalculator");
  audio_header >> audio_converter_node.SideIn("IN_AUDIO_HEADER");
  graph.SideIn("OUT_AUDIO_HEADER")
          .SetName("out_audio_header")
          .Cast<aikit::media::AudioStreamParameters>() >>
      audio_converter_node.SideIn("OUT_AUDIO_HEADER");
  audio_stream >> audio_converter_node.In("IN_AUDIO");
  auto float_48kHz_audio_stream = audio_converter_node.Out("OUT_AUDIO");

  // Processing

  // audio
  auto &audio_subgraph = graph.AddNode("AudioGraph");
  audio_header >> audio_subgraph.SideIn("IN_AUDIO_HEADER");
  audio_stream >> audio_subgraph.In("IN_AUDIO");
    graph.SideIn("OUT_ASR_AUDIO_HEADER")
          .SetName("out_asr_audio_header")
          .Cast<aikit::media::AudioStreamParameters>() >>
      audio_subgraph.SideIn("OUT_AUDIO_HEADER");

  graph.SideIn("ASR_MODEL_PATH")
          .SetName("asr_model_path")
          .Cast<std::string>() >>
      audio_subgraph.SideIn("ASR_MODEL_PATH");
  graph.SideIn("SPK_MODEL_PATH")
          .SetName("spk_model_path")
          .Cast<std::string>() >>
      audio_subgraph.SideIn("SPK_MODEL_PATH");
  auto transcription_stream = audio_subgraph.Out("TRANSCRIPTION");

  // visual
  auto &visual_subgraph = graph.AddNode("VisualGraph");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      visual_subgraph.SideIn("OUT_VIDEO_HEADER");
  graph.SideIn("DETECTION_MODEL_PATH")
          .SetName("detection_model_path")
          .Cast<std::string>() >>
      visual_subgraph.SideIn("DETECTION_MODEL_PATH");
  graph.SideIn("OCR_MODEL_PATH")
          .SetName("ocr_model_path")
          .Cast<std::string>() >>
      visual_subgraph.SideIn("OCR_MODEL_PATH");
  yuv_video_stream >> visual_subgraph.In("IN_VIDEO");
  auto detections_stream = visual_subgraph.Out("DETECTIONS");
  auto speaker_name_stream = visual_subgraph.Out("STRING");

  // End of processing

  // Send to Evaluator
  auto &evaluator_client_node = graph.AddNode("EvaluatorClientCalculator");
  detections_stream >> evaluator_client_node.In("DETECTIONS");
  speaker_name_stream >> evaluator_client_node.In("SPEAKER_NAME");
  transcription_stream >> evaluator_client_node.In("ASR_RESULT");

  // Write audio
  auto &sink_video_node = graph.AddNode("FFMPEGSinkVideoCalculator");
  graph.SideIn("OUTPUT_FILE_PATH")
          .SetName("output_file_path")
          .Cast<std::string>() >>
      sink_video_node.SideIn("OUTPUT_FILE_PATH");
  graph.SideIn("OUT_AUDIO_HEADER")
          .SetName("out_audio_header")
          .Cast<aikit::media::AudioStreamParameters>() >>
      sink_video_node.SideIn("AUDIO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      sink_video_node.SideIn("VIDEO_HEADER");
  float_48kHz_audio_stream >> sink_video_node.In("AUDIO");
  yuv_video_stream >> sink_video_node.In("VIDEO");

  return graph.GetConfig();
}

absl::Status RunMPPGraph() {
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  auto config = BuildGraph();

  std::map<std::string, mediapipe::Packet> input_side_packets;
  input_side_packets["output_file_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_output_file_path));

  aikit::media::AudioStreamParameters audio_stream_parameters;
  input_side_packets["out_audio_header"] =
      mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
          audio_stream_parameters);

  aikit::media::AudioStreamParameters asr_audio_stream_parameters;
  asr_audio_stream_parameters.sample_rate = 16000;
  input_side_packets["out_asr_audio_header"] =
      mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
          asr_audio_stream_parameters);

  aikit::media::VideoStreamParameters video_stream_parameters;
  input_side_packets["out_video_header"] =
      mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
          video_stream_parameters);
  input_side_packets["detection_model_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_cdetr_model_path));
  input_side_packets["ocr_model_path"] =
        mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_ocr_model_path));

  input_side_packets["asr_model_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_asr_model_path));
  input_side_packets["spk_model_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_spk_model_path));

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  std::unique_lock<std::mutex> lock(SIGNAL_STATUS_MUTEX);
  SIGNAL_STATUS_COND_V.wait(lock, [&] {
    return (SIGNAL_STATUS == SIGINT || SIGNAL_STATUS == SIGTERM);
  });
  ABSL_LOG(INFO) << "Start closing all packet sources of the calculator graph.";
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  ABSL_LOG(INFO) << "Waiting until done...";
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

int main(int argc, char **argv) {
  absl::SetProgramUsageMessage(
      "Captures screen/audio of a machine and process them.");
  absl::ParseCommandLine(argc, argv);

  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }

  return EXIT_SUCCESS;
}
