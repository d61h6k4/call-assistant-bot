
#include "mediapipe/calculators/core/packet_thinner_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "mediapipe/framework/subgraph.h"
#include <string_view>

#include "av_transducer/utils/audio.h"
#include "av_transducer/utils/video.h"

namespace aikit {
// A VisualGraph performs extraction of data from video (images).
//
// Inputs:
//   VIDEO - media::VideoFrame
//     Image (stream of images, so video) to extract thumbnails from
// Outputs:
//   Detections - vector of detections
class VisualGraph : public mediapipe::Subgraph {
public:
  static constexpr std::string_view kInVideo = "IN_VIDEO";
  static constexpr std::string_view kOutDetections = "DETECTIONS";

  absl::StatusOr<mediapipe::CalculatorGraphConfig>
  GetConfig(mediapipe::SubgraphContext *sc) override {
    mediapipe::api2::builder::Graph graph;

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
    graph.In(kInVideo) >> packet_thinner_node.In("");
    auto resampled_video_stream =
        packet_thinner_node.Out("").Cast<media::VideoFrame>();

    // Lift to Mediapipe YUVImage
    auto &lift_yuvimage_node = graph.AddNode("LiftToYUVImageCalculator");
    graph.SideIn("OUT_VIDEO_HEADER")
            .SetName("out_video_header")
            .Cast<media::VideoStreamParameters>() >>
        lift_yuvimage_node.SideIn("IN_VIDEO_HEADER");
    resampled_video_stream >> lift_yuvimage_node.In("IN_VIDEO");
    auto yuvimage_stream = lift_yuvimage_node.Out("OUT_VIDEO");

    // YUV to Image (RGB)
    auto &yuv_to_image_node = graph.AddNode("YUVToImageCalculator");
    yuvimage_stream >> yuv_to_image_node.In("YUV_IMAGE");
    auto images_stream = yuv_to_image_node.Out("IMAGE");

    // Apply CDETR
    auto &cdetr_node = graph.AddNode("CDETRCalculator");
    graph.SideIn("CDETR_MODEL_PATH")
            .SetName("cdetr_model_path")
            .Cast<std::string>() >>
        cdetr_node.SideIn("CDETR_MODEL_PATH");
    images_stream >> cdetr_node.In("IMAGE");
    cdetr_node.Out("DETECTIONS") >> graph.Out(kOutDetections);

    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(VisualGraph);
} // namespace aikit
