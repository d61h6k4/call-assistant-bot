#include <vosk_api.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "ml/asr/model.h"


int main() {
    int16_t BUFSIZE = 32000;
    std::vector<float> buf(BUFSIZE);
    int nread, final;

    auto model = aikit::ml::ASRModel("ml/asr/models/vosk-model-ru-0.22", "ml/asr/models/vosk-model-spk-0.4");

    std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary);
    wavin.seekg(44);
    while (wavin) {
        wavin.read(reinterpret_cast<char*>(buf.data()), BUFSIZE * sizeof(float));
        nread = wavin.gcount() / sizeof(float);
        for(int i = 0; i < BUFSIZE; i++) {
            buf[i] *= 32767.0f;
        }
        auto result = model(buf);
        if (result.is_final) {
            std::string text = result.text;
            printf("Распознанный текст: %s\n", text.c_str());
            std::vector<float> spk_vector = result.spk_embedding;
            printf("Вектор характеристик спикера (первые 5 элементов): ");
            for (int i = 0; i < 5 && i < spk_vector.size(); i++) {
                printf("%.6f ", spk_vector[i]);
            }
            printf("\n");
        }
    }
    return 0;
}