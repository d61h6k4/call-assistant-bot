#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/absl_log.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_graph.h"
#include <string>

ABSL_FLAG(std::string, input_file_path, "", "Full path of video to read.");

mediapipe::CalculatorGraphConfig BuildGraph() {
  mediapipe::api2::builder::Graph graph;

  // Capture video device
  auto &source_video_node = graph.AddNode("FFMPEGSourceVideoCalculator");
  graph.SideIn("INPUT_FILE_PATH")
          .SetName("input_file_path")
          .Cast<std::string>() >>
      source_video_node.SideIn("INPUT_FILE_PATH");
  auto video_header = source_video_node.SideOut("VIDEO_HEADER");
  auto audio_header = source_video_node.SideOut("AUDIO_HEADER");
  auto audio_stream = source_video_node.Out("AUDIO");
  auto video_stream = source_video_node.Out("VIDEO");

  // Convert to YUV420P
  auto &video_converter_node = graph.AddNode("VideoConverterCalculator");
  video_header >> video_converter_node.SideIn("IN_VIDEO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      video_converter_node.SideIn("OUT_VIDEO_HEADER");
  video_stream >> video_converter_node.In("IN_VIDEO");
  auto yuv_video_stream = video_converter_node.Out("OUT_VIDEO");

  auto &visual_subgraph = graph.AddNode("VisualGraph");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      visual_subgraph.SideIn("OUT_VIDEO_HEADER");
  graph.SideIn("CDETR_MODEL_PATH")
          .SetName("cdetr_model_path")
          .Cast<std::string>() >>
      visual_subgraph.SideIn("CDETR_MODEL_PATH");
  graph.SideIn("OCR_MODEL_PATH")
          .SetName("ocr_model_path")
          .Cast<std::string>() >>
      visual_subgraph.SideIn("OCR_MODEL_PATH");
  yuv_video_stream >> visual_subgraph.In("IN_VIDEO");
  auto detections_stream = visual_subgraph.Out("DETECTIONS");
  auto speaker_name_stream = visual_subgraph.Out("STRING");

  // Dump to stdout
  auto &dumper_node = graph.AddNode("DumperCalculator");
  detections_stream >> dumper_node.In("DETECTIONS");
  speaker_name_stream >> dumper_node.In("STRING");

  return graph.GetConfig();
}

absl::Status RunMPPGraph() {
  auto config = BuildGraph();

  std::map<std::string, mediapipe::Packet> input_side_packets;
  input_side_packets["input_file_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_input_file_path));

  aikit::media::VideoStreamParameters video_stream_parameters;
  input_side_packets["out_video_header"] =
      mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
          video_stream_parameters);
  input_side_packets["cdetr_model_path"] =
      mediapipe::MakePacket<std::string>("ml/detection/models/model.onnx");
  input_side_packets["ocr_model_path"] =
      mediapipe::MakePacket<std::string>("ml/ocr/models/model.onnx");

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));
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
