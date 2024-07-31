#pragma once
#include <string>
#include <vector>
#include <memory>
#include "absl/status/statusor.h"
#include "vosk_api.h"

namespace aikit::ml {

struct ASRResult {
    std::string text;
    std::vector<float> spk_embedding;
};


class ASRModel {
public:
  ASRModel(const std::string& model_path, const std::string& spk_model_path, size_t sample_rate = 16000);
  ~ASRModel();

  ASRModel(ASRModel&& other) noexcept;

  ASRModel& operator=(ASRModel&& other) noexcept;

  ASRModel(const ASRModel&) = delete;
  ASRModel& operator=(const ASRModel&) = delete;

  absl::StatusOr<ASRResult> operator()(const std::vector<float>& audio_buffer);
private:
  const std::string log_id_ = "asr_model";

  std::string model_path_;
  std::string spk_model_path_;
  size_t sample_rate_;
  std::unique_ptr<VoskModel, void(*)(VoskModel*)> model_;
  std::unique_ptr<VoskSpkModel, void(*)(VoskSpkModel*)> spk_model_;
  std::unique_ptr<VoskRecognizer, void(*)(VoskRecognizer*)> recognizer_;
  void initialize();
};
}  // namespace aikit::ml
