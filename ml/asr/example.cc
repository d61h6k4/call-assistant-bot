#include <vosk_api.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>


int main() {
    int16_t BUFSIZE = 32000;
    std::vector<float> buf(BUFSIZE);
    int nread, final;

    VoskModel *model = vosk_model_new("ml/asr/models/vosk-model-ru-0.22");
    VoskSpkModel *spk_model = vosk_spk_model_new("ml/asr/models/vosk-model-spk-0.4");
    VoskRecognizer *recognizer = vosk_recognizer_new_spk(model, 16000.0, spk_model);

    std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary);
    wavin.seekg(44);
    while (wavin) {
        wavin.read(reinterpret_cast<char*>(buf.data()), BUFSIZE * sizeof(float));
        nread = wavin.gcount() / sizeof(float);
        for(int i = 0; i < BUFSIZE; i++) {
            buf[i] *= 32767.0f;
        }
         final = vosk_recognizer_accept_waveform_f(recognizer, buf.data(), nread);
         if (final) {
            const char* json_result = vosk_recognizer_result(recognizer);
            nlohmann::json result_json = nlohmann::json::parse(json_result);

            if (result_json.contains("text")) {
                std::string text = result_json["text"];
                printf("Распознанный текст: %s\n", text.c_str());
            }

            if (result_json.contains("spk_frames")) {
                int spk_frames = result_json["spk_frames"];
                printf("Количество кадров спикера: %d\n", spk_frames);
            }
            if (result_json.contains("spk")) {
                std::vector<float> spk_vector = result_json["spk"].get<std::vector<float>>();
                printf("Вектор характеристик спикера (первые 5 элементов): ");
                for (int i = 0; i < 5 && i < spk_vector.size(); i++) {
                    printf("%.6f ", spk_vector[i]);
                }
                printf("\n");
            }
        }
    }
    vosk_recognizer_free(recognizer);
    vosk_spk_model_free(spk_model);
    vosk_model_free(model);
    return 0;
}