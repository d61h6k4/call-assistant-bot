

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/absl_log.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/calculators/core/packet_thinner_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/yuv_image.h"

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

  // Create 1 FPS stream
  auto &packet_thinner_node = graph.AddNode("PacketThinnerCalculator");
  auto &packet_thinner_node_opts =
      packet_thinner_node
          .GetOptions<mediapipe::PacketThinnerCalculatorOptions>();
  // Period controls how frequently we want to take a new packets (in
  // microseconds) 1 FPS is 1 frame in 1 seconds
  packet_thinner_node_opts.set_period(1000000);
  packet_thinner_node_opts.set_thinner_type(
      mediapipe::PacketThinnerCalculatorOptions::ASYNC);
  video_stream >> packet_thinner_node.In("");
  auto resampled_video_stream =
      packet_thinner_node.Out("").Cast<aikit::media::VideoFrame>();

  // Convert to YUV420P
  auto &video_converter_node = graph.AddNode("VideoConverterCalculator");
  video_header >> video_converter_node.SideIn("IN_VIDEO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      video_converter_node.SideIn("OUT_VIDEO_HEADER");
  resampled_video_stream >> video_converter_node.In("IN_VIDEO");
  auto yuv_video_stream = video_converter_node.Out("OUT_VIDEO");

  // Lift to Mediapipe YUVImage
  auto &lift_yuvimage_node = graph.AddNode("LiftToYUVImageCalculator");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      lift_yuvimage_node.SideIn("IN_VIDEO_HEADER");
  yuv_video_stream >> lift_yuvimage_node.In("IN_VIDEO");
  auto yuvimage_stream = lift_yuvimage_node.Out("OUT_VIDEO");

  // YUV to Image (RGB)
  auto &yuv_to_image_node = graph.AddNode("YUVToImageCalculator");
  yuvimage_stream >> yuv_to_image_node.In("YUV_IMAGE");
  auto images_stream = yuv_to_image_node.Out("IMAGE");

  // Apply CDETR
  auto &cdetr_node = graph.AddNode("CDETRCalculator");
  images_stream >> cdetr_node.In("IMAGE");
  auto detections_stream = cdetr_node.Out("DETECTIONS");

  // Dump to stdout
  auto &dumper_node = graph.AddNode("DumperCalculator");
  detections_stream >> dumper_node.In("DETECTIONS");

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
