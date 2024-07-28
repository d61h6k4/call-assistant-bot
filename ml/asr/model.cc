#include <iostream>
#include <stdexcept>

#include "ml/asr/model.h"
#include "json.h"

namespace aikit::ml {

ASRModel::ASRModel(const std::string& model_path, const std::string& spk_model_path, size_t sample_rate)
    : model_path_(model_path), spk_model_path_(spk_model_path), sample_rate_(sample_rate),
      model_(nullptr, vosk_model_free), spk_model_(nullptr, vosk_spk_model_free), recognizer_(nullptr, vosk_recognizer_free) {
    initialize();
}

ASRModel::~ASRModel() = default;

ASRModel::ASRModel(ASRModel&& other) noexcept
    : model_path_(std::move(other.model_path_)),
      spk_model_path_(std::move(other.spk_model_path_)),
      sample_rate_(other.sample_rate_),
      model_(std::move(other.model_)),
      spk_model_(std::move(other.spk_model_)),
      recognizer_(std::move(other.recognizer_)) {}

ASRModel& ASRModel::operator=(ASRModel&& other) noexcept {
    if (this != &other) {
        model_path_ = std::move(other.model_path_);
        spk_model_path_ = std::move(other.spk_model_path_);
        sample_rate_ = other.sample_rate_;
        model_ = std::move(other.model_);
        spk_model_ = std::move(other.spk_model_);
        recognizer_ = std::move(other.recognizer_);
    }
    return *this;
}

void ASRModel::initialize() {
    model_.reset(vosk_model_new(model_path_.c_str()));
    if (!model_) {
        throw std::runtime_error("Failed to create model");
    }

    spk_model_.reset(vosk_spk_model_new(spk_model_path_.c_str()));
    if (!spk_model_) {
        throw std::runtime_error("Failed to create speaker model");
    }

    recognizer_.reset(vosk_recognizer_new_spk(model_.get(), sample_rate_, spk_model_.get()));
    if (!recognizer_) {
        throw std::runtime_error("Failed to create recognizer");
    }
}

absl::StatusOr<ASRResult> ASRModel::operator()(const std::vector<float>& audio_buffer) {
    int final_status = vosk_recognizer_accept_waveform_f(recognizer_.get(), audio_buffer.data(), audio_buffer.size());

    if (final_status != 0) {
        std::string result_str = vosk_recognizer_result(recognizer_.get());
        ASRResult result;
        try {
            auto json_result = json::JSON::Load(result_str);
            if (json_result.hasKey("text") && json_result.hasKey("spk")) {
                result.text = json_result["text"].ToString();
                for (auto& item : json_result["spk"].ArrayRange()) {
                    result.spk_embedding.push_back(item.ToFloat());
                }
                return result;
            }
        } catch (const std::exception& e) {
            return absl::InternalError("Error parsing JSON");
        }
    }
    return absl::UnavailableError("Result not ready");
}

}  // namespace aikit::ml
