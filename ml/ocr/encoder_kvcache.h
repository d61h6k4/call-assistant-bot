#pragma once

#include <memory>
#include "models/onnxruntime_api.h"

namespace Generators {
struct EncoderKVCache {
  EncoderKVCache(const Model& model, State& state);

  void AddEncoder();  // If model has an initial encoder step, this is used
  void Update();

 private:
  const Model& model_;
  State& state_;
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};
  bool past_present_share_buffer_;  // True if model.decoder.past_present_share_buffer is set to true, and we're using cuda, and not beam search

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};
}  // namespace Generators
