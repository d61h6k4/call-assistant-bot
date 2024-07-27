#include "ml/asr/model.h"
#include <iostream>
#include <stdexcept>

namespace aikit::ml {

ASRModel::ASRModel(const std::string& model_path, const std::string& spk_model_path)
    : model_path_(model_path), spk_model_path_(spk_model_path), model_(nullptr), spk_model_(nullptr), recognizer_(nullptr) {
    initialize();
}

ASRModel::~ASRModel() {
    if (recognizer_) vosk_recognizer_free(recognizer_);
    if (spk_model_) vosk_spk_model_free(spk_model_);
    if (model_) vosk_model_free(model_) ;
}

void ASRModel::initialize() {
    model_ = vosk_model_new(model_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Не удалось создать модель");
    }

    spk_model_ = vosk_spk_model_new(spk_model_path_.c_str());
    if (!spk_model_) {
        throw std::runtime_error("Не удалось создать модель диктора");
    }

    recognizer_ = vosk_recognizer_new_spk(model_, sample_rate, spk_model_);
    if (!recognizer_) {
        throw std::runtime_error("Не удалось создать распознаватель");
    }
}

ASRResult ASRModel::operator()(const std::vector<float>& audio_buffer) {
    ASRResult result;
    result.is_final = false;
    int final = vosk_recognizer_accept_waveform_f(recognizer_, audio_buffer.data(), audio_buffer.size());

    if (final) {
        const char* result_str = vosk_recognizer_result(recognizer_);
        try {
            auto json_result = nlohmann::json::parse(result_str);

            if (json_result.contains("text") && json_result.contains("spk")) {
                result.is_final = true;
                result.text = json_result["text"].get<std::string>();
                result.spk_embedding = json_result["spk"].get<std::vector<float>>();
                return result;
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << log_id_ << ": Ошибка парсинга JSON: " << e.what() << std::endl;
        }
    }

    return result;
}

}  // namespace aikit::ml
