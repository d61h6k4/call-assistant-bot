
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/subgraph.h"
#include <string_view>

#include "mediapipe/calculators/image/scale_image_calculator.pb.h"

namespace aikit {
// An OCRGraph performs extraction of text from a given area on frames (images).
//
// Inputs:
//   Image - media::Image
//    Rect - region of interest
// Outputs:
//   String - text in area
class OCRGraph : public mediapipe::Subgraph {
public:
  static constexpr std::string_view kInVideo = "IMAGE";
  static constexpr std::string_view kInROI = "NORM_RECT";
  static constexpr std::string_view kOutText = "STRING";

  absl::StatusOr<mediapipe::CalculatorGraphConfig>
  GetConfig(mediapipe::SubgraphContext *sc) override {
    mediapipe::api2::builder::Graph graph;

    // Convert Image to ImageFrame (built in calculators expect ImageFrame)
    auto &from_image_node = graph.AddNode("FromImageCalculator");
    graph.In(kInVideo) >> from_image_node.In("IMAGE");
    auto image_frame_stream = from_image_node.Out("IMAGE_CPU");

    // Crop region of interest
    auto &image_cropping_node = graph.AddNode("ImageCroppingCalculator");
    image_frame_stream >> image_cropping_node.In("IMAGE");
    graph.In(kInROI) >> image_cropping_node.In("NORM_RECT");
    auto cropped_image_frame_stream = image_cropping_node.Out("IMAGE");

    // Scale image to 64x256 (OCR model expects this size)
    auto &scale_image_node = graph.AddNode("ScaleImageCalculator");
    auto &scale_image_options =
        scale_image_node.GetOptions<mediapipe::ScaleImageCalculatorOptions>();
    scale_image_options.set_target_width(256);
    scale_image_options.set_target_height(64);
    scale_image_options.set_min_aspect_ratio("0/1");
    scale_image_options.set_max_aspect_ratio("0/1");
    scale_image_options.set_scale_to_multiple_of(1);
    scale_image_options.set_preserve_aspect_ratio(false);
    scale_image_options.set_algorithm(
        ::mediapipe::ScaleImageCalculatorOptions_ScaleAlgorithm::
            ScaleImageCalculatorOptions_ScaleAlgorithm_CUBIC);
    cropped_image_frame_stream >> scale_image_node.In("FRAMES");
    auto scaled_image_frame_stream = scale_image_node.Out("FRAMES");

    auto &ocr_node = graph.AddNode("OCRCalculator");
    graph.SideIn("OCR_MODEL_PATH")
            .SetName("ocr_model_path")
            .Cast<std::string>() >>
        ocr_node.SideIn("OCR_MODEL_PATH");
    scaled_image_frame_stream >> ocr_node.In("IMAGE_FRAME");
    ocr_node.Out("STRING") >> graph.Out(kOutText);

    return graph.GetConfig();
  }
};
REGISTER_MEDIAPIPE_GRAPH(OCRGraph);
} // namespace aikit
