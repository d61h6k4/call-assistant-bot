
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_graph.h"

ABSL_FLAG(std::string, output_file_path, "", "Full path of video to save.");

mediapipe::CalculatorGraphConfig BuildGraph() {
  mediapipe::api2::builder::Graph graph;

  // Capture audio device
  auto &capture_audio_node = graph.AddNode("FFMPEGCaptureAudioCalculator");
  auto audio_header = capture_audio_node.SideOut("AUDIO_HEADER");
  auto audio_stream = capture_audio_node.Out("AUDIO");

  // Write audio
  auto &sink_video_node = graph.AddNode("FFMPEGSinkVideoCalculator");
  graph.SideIn("OUTPUT_FILE_PATH")
          .SetName("output_file_path")
          .Cast<std::string>() >>
      sink_video_node.SideIn("OUTPUT_FILE_PATH");
  audio_header >> sink_video_node.SideIn("AUDIO_HEADER");
  audio_stream >> sink_video_node.In("AUDIO");

  return graph.GetConfig();
}

absl::Status RunMPPGraph() {
  auto config = BuildGraph();

  std::map<std::string, mediapipe::Packet> input_side_packets;
  input_side_packets["output_file_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_output_file_path));

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
