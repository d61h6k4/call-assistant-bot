
#include "models/florence2/processor.h"
#include <cstdint>

namespace aikit {

namespace {

std::unique_ptr<OrtValue> ProcessPrompt(const Generators::Tokenizer &tokenizer,
                                        const std::string &prompt,
                                        Ort::Allocator &allocator) {

  std::vector<int32_t> input_ids = tokenizer.Encode(prompt.c_str());

  // input_ids is created. Pack it into an allocated OrtValue to avoid managing
  // the memory.
  const std::vector<int64_t> shape{1, static_cast<int64_t>(input_ids.size())};
  auto input_ids_value = OrtValue::CreateTensor<int32_t>(allocator, shape);
  std::copy(input_ids.begin(), input_ids.end(),
            input_ids_value->GetTensorMutableData<int32_t>());
  return input_ids_value;
}

std::unique_ptr<OrtValue>
ProcessPixelValues(const Generators::Images *pixel_values,
                   Ort::Allocator &allocator) {

  const std::vector<int64_t> shape{
      static_cast<int64_t>(pixel_values->images_.get()->size())};
  auto pixel_values_value = OrtValue::CreateTensor<uint8_t>(allocator, shape);
  std::copy(pixel_values->images_.get()->begin(),
            pixel_values->images_.get()->end(),
            pixel_values_value->GetTensorMutableData<uint8_t>());

  return pixel_values_value;
}
} // namespace

Processor::Processor(Generators::Config &config,
                     const Generators::SessionInfo &session_info)
    : pixel_values_type_(session_info.GetInputDataType(
          config.model.vision.inputs.pixel_values)) {

  config.AddMapping(std::string(Generators::Config::Defaults::InputIdsName),
                    config.model.embedding.inputs.input_ids);
  config.AddMapping(std::string(Generators::Config::Defaults::PixelValuesName),
                    config.model.vision.inputs.pixel_values);
}

std::unique_ptr<Generators::NamedTensors>
Processor::Process(const Generators::Tokenizer &tokenizer,
                   const std::string &prompt,
                   const Generators::Images *images) {
  Ort::Allocator &allocator{Ort::Allocator::GetWithDefaultOptions()};
  auto named_tensors = std::make_unique<Generators::NamedTensors>();

  named_tensors->emplace(
      std::string(Generators::Config::Defaults::InputIdsName),
      std::make_shared<Generators::Tensor>(
          ProcessPrompt(tokenizer, prompt, allocator)));
  named_tensors->emplace(
      std::string(Generators::Config::Defaults::PixelValuesName),
      std::make_shared<Generators::Tensor>(
          ProcessPixelValues(images, allocator)));

  return named_tensors;
}
}; // namespace aikit
