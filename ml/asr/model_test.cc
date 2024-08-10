
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <fstream>

#include "ml/asr/model.h"

#include "absl/log/absl_log.h"
#include <chrono>


TEST(TestMLASRModel, SanityCheck) {
    auto model = aikit::ml::ASRModel("ml/asr/models/vosk-model-ru-0.22", "ml/asr/models/vosk-model-spk-0.4");
    std::string text;
    std::int32_t size_emb;

    std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary | std::ios::ate);
    std::streamsize size = 16000 * 10;
    wavin.seekg(44, std::ios::beg);
    std::vector<float> audio_buffer(size);
    wavin.read(reinterpret_cast<char*>(audio_buffer.data()), size * sizeof(float));
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = model(audio_buffer);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_TRUE(result.ok());
    ABSL_LOG(INFO) << "Time duration: " << duration.count() / 1000.0 << " seconds";
    text = result.value().text;
    size_emb = result.value().spk_embedding.size();
    ABSL_LOG(INFO) << text << "\n";
    EXPECT_EQ(size_emb, 128);
}
