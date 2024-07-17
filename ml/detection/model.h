#pragma once

#include <array>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <vector>

namespace aikit::ml {
// A bounding box. The box is defined by its upper left corner (xmin, ymin)
// and its width and height, all in coordinates normalized by the image
// dimensions.
struct Detection {
  float xmin;
  float ymin;
  float width;
  float height;
  int label_id;
  float score;
};

class CDetr {
public:
  CDetr();

  std::vector<Detection> operator()(const uint8_t *image);

public:
  static constexpr size_t width = 1280;
  static constexpr size_t height = 720;

private:
  std::string log_id_ = "cdetr";
  OrtLoggingLevel logging_level_ = ORT_LOGGING_LEVEL_WARNING;

  Ort::Env env_;
  Ort::RunOptions run_options_;
  Ort::SessionOptions session_options_;
  Ort::Session session_{nullptr};

  Ort::Allocator allocator_device_{nullptr};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 3> input_shape_{height, width, 3};

  static constexpr std::array<const char *, 1> input_names_ = {"image"};
  static constexpr std::array<const char *, 1> output_names_ = {"nms_out"};
};
} // namespace aikit::ml
