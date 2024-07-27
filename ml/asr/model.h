#pragma once

#include <vosk_api.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>


namespace aikit::ml {

struct ASRResult {
    bool is_final;
    std::string text;
    std::vector<float> spk_embedding;
};


class ASRModel {
public:
  explicit ASRModel(const std::string& model_path, const std::string& spk_model_path);
  ~ASRModel();

  ASRResult operator()(const std::vector<float>& audio_buffer);

public:
  static constexpr size_t sample_rate = 16000;
  static constexpr size_t channels = 1;

private:
  std::string log_id_ = "asr_model";

  std::string model_path_;
  std::string spk_model_path_;
  VoskModel *model_;
  VoskSpkModel *spk_model_;
  VoskRecognizer *recognizer_;

  void initialize();
};
}  // namespace aikit::ml
