#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/absl_log.h"
#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"
#include "mediapipe/calculators/util/detections_to_render_data_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/util/color.pb.h"
#include <string>

ABSL_FLAG(std::string, input_file_path, "", "Full path of video to read.");
ABSL_FLAG(std::string, output_file_path, "", "Full path of video to save.");
ABSL_FLAG(bool, profile, false, "Full path of video to save.");

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
  auto transcription = audio_subgraph.Out("TRANSCRIPTION");

  // Debug graph
  auto &det_to_render_node = graph.AddNode("DetectionsToRenderDataCalculator");
  auto &det_to_render_options =
      det_to_render_node
          .GetOptions<mediapipe::DetectionsToRenderDataCalculatorOptions>();
  det_to_render_options.set_thickness(1.0);
  auto *det_color = det_to_render_options.mutable_color();
  det_color->set_r(229);
  det_color->set_g(75);
  det_color->set_b(75);
  detections_stream >> det_to_render_node.In("DETECTIONS");
  auto det_render_data_stream = det_to_render_node.Out("RENDER_DATA");

  auto &speaker_to_render_node = graph.AddNode("SpeakerNameToRenderCalculator");
  speaker_name_stream >> speaker_to_render_node.In("SPEAKER_NAME");
  auto speaker_render_data_stream = speaker_to_render_node.Out("RENDER_DATA");

  auto &packet_cloner_node = graph.AddNode("PacketClonerCalculator");
  yuv_video_stream >> packet_cloner_node.In("TICK");
  det_render_data_stream >> packet_cloner_node.In(0);
  speaker_render_data_stream >> packet_cloner_node.In(1);
  auto cloned_det_render_data_stream = packet_cloner_node.Out(0);
  auto cloned_speaker_render_data_stream = packet_cloner_node.Out(1);

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

  // Convert Image to ImageFrame (built in calculators expect ImageFrame)
  auto &from_image_node = graph.AddNode("FromImageCalculator");
  images_stream >> from_image_node.In("IMAGE");
  auto image_frame_stream = from_image_node.Out("IMAGE_CPU");

  auto &overlay_node = graph.AddNode("AnnotationOverlayCalculator");
  cloned_det_render_data_stream >> overlay_node.In(0);
  cloned_speaker_render_data_stream >> overlay_node.In(1);
  image_frame_stream >> overlay_node.In("IMAGE");

  auto annotated_images_stream = overlay_node.Out("IMAGE");

  auto &to_video_frame_node = graph.AddNode("ImageFrameToVideoFrameCalculator");
  annotated_images_stream >> to_video_frame_node.In("IMAGE_FRAME");
  auto res_video_header = to_video_frame_node.SideOut("OUT_VIDEO_HEADER");
  auto video_frame_stream = to_video_frame_node.Out("VIDEO_FRAME");

  // Write audio
  auto &video_converter_back_node = graph.AddNode("VideoConverterCalculator");
  res_video_header >> video_converter_back_node.SideIn("IN_VIDEO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      video_converter_back_node.SideIn("OUT_VIDEO_HEADER");
  video_frame_stream >> video_converter_back_node.In("IN_VIDEO");
  auto back_yuv_video_stream = video_converter_back_node.Out("OUT_VIDEO");

  auto &sink_video_node = graph.AddNode("FFMPEGSinkVideoCalculator");
  graph.SideIn("OUTPUT_FILE_PATH")
          .SetName("output_file_path")
          .Cast<std::string>() >>
      sink_video_node.SideIn("OUTPUT_FILE_PATH");
  audio_header >> sink_video_node.SideIn("AUDIO_HEADER");
  graph.SideIn("OUT_VIDEO_HEADER")
          .SetName("out_video_header")
          .Cast<aikit::media::VideoStreamParameters>() >>
      sink_video_node.SideIn("VIDEO_HEADER");
  audio_stream >> sink_video_node.In("AUDIO");
  back_yuv_video_stream >> sink_video_node.In("VIDEO");

  return graph.GetConfig();
}

absl::Status RunMPPGraph() {
  auto config = BuildGraph();

  std::map<std::string, mediapipe::Packet> input_side_packets;

  aikit::media::AudioStreamParameters asr_audio_stream_parameters;
  asr_audio_stream_parameters.sample_rate = 16000;
  input_side_packets["out_asr_audio_header"] =
      mediapipe::MakePacket<aikit::media::AudioStreamParameters>(
          asr_audio_stream_parameters);

  input_side_packets["input_file_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_input_file_path));
  input_side_packets["output_file_path"] =
      mediapipe::MakePacket<std::string>(absl::GetFlag(FLAGS_output_file_path));
  aikit::media::VideoStreamParameters video_stream_parameters;
  input_side_packets["out_video_header"] =
      mediapipe::MakePacket<aikit::media::VideoStreamParameters>(
          video_stream_parameters);
  input_side_packets["detection_model_path"] =
      mediapipe::MakePacket<std::string>("ml/detection/models/model.onnx");
  input_side_packets["ocr_model_path"] =
      mediapipe::MakePacket<std::string>("ml/ocr/models/model.onnx");

  input_side_packets["asr_model_path"] =
      mediapipe::MakePacket<std::string>("ml/asr/models/vosk-model-ru-0.42");
  input_side_packets["spk_model_path"] =
      mediapipe::MakePacket<std::string>("ml/asr/models/vosk-model-spk-0.4");

  if (absl::GetFlag(FLAGS_profile)) {
    // Enable profiling
    mediapipe::ProfilerConfig *profilerConfig =
        config.mutable_profiler_config();
    profilerConfig->set_trace_enabled(true);
    profilerConfig->set_enable_profiler(true);
    profilerConfig->set_trace_log_disabled(false);
  }

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
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
