
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <fstream>
#include "ml/asr/model.h"

#include "absl/log/absl_log.h"


TEST(TestMLASRModel, SanityCheck) {
    int16_t BUFFER_SIZE = 32000;
    auto model = aikit::ml::ASRModel("ml/asr/models/vosk-model-ru-0.22", "ml/asr/models/vosk-model-spk-0.4");

    std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary);
    wavin.seekg(44);
    std::vector<float> audio_buffer(BUFFER_SIZE);
    while (wavin) {
        wavin.read(reinterpret_cast<char*>(buf.data()), BUFSIZE * sizeof(float));
        nread = wavin.gcount() / sizeof(float);
        for(int i = 0; i < nread; i++) {
            buf[i] *= 32767.0f;
        }
        auto result = model(audio_buffer);
        if (result.is_final) {
            ABSL_LOG(INFO) << result.text << "\n";
            EXPECT_EQ(result.spk_embedding.size(), 128);
        }
    }
}