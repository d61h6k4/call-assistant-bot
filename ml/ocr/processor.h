
#pragma once

#include "image_processor.h"
#include "models/model.h"
#include "models/utils.h"
#include "ortx_processor.h"
#include <cstddef>

namespace aikit {

class Processor {
public:
  Processor(Generators::Config &config,
            const Generators::SessionInfo &session_info);

  // Returned NamedTensors own the OrtValue and are not owned by the caller.
  // OrtValue memory will be released when the NamedTensors are destroyed.
  std::unique_ptr<Generators::NamedTensors>
  Process(const Generators::Tokenizer &tokenizer, const std::string &prompt,
          const uint8_t *images);

public:
  static constexpr int64_t width = 768;
  static constexpr int64_t height = 768;

private:
  std::string input_ids_name_;
  std::string pixel_values_name_;
  ONNXTensorElementDataType pixel_values_type_;
};
}; // namespace aikit
