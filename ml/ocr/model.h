#pragma once

#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <string>

namespace aikit::ml {
class OCR {
public:
  explicit OCR(const std::string &path_to_model);
  std::string operator()(const uint8_t *image);

public:
  static constexpr int64_t height = 64;
  static constexpr int64_t width = 256;

private:
  std::string log_id_ = "ocr_easyocr";
  OrtLoggingLevel logging_level_ = ORT_LOGGING_LEVEL_WARNING;

  Ort::Env env_;
  Ort::RunOptions run_options_;
  Ort::SessionOptions session_options_;
  Ort::Session session_{nullptr};

  Ort::Allocator allocator_device_{nullptr};

  std::array<int64_t, 2> input_shape_{height, width};
  std::array<int64_t, 2> output_shape_{1, 63};

  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};

  static constexpr std::array<const char *, 1> input_names_ = {"image"};
  static constexpr std::array<const char *, 1> output_names_ = {"index"};

};

} // namespace aikit::ml
