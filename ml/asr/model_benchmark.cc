#include "benchmark/benchmark.h"

#include <vector>
#include <cstdint>
#include <fstream>
#include <memory>

#include "ml/asr/model.h"

std::unique_ptr<aikit::ml::ASRModel> g_model;
std::vector<float> g_audio_buffer;


void SetupBenchmark(benchmark::State& state) {
    g_model = std::make_unique<aikit::ml::ASRModel>("ml/asr/models/vosk-model-ru-0.22", "ml/asr/models/vosk-model-spk-0.4");
    
    std::ifstream wavin("testdata/meeting_audio.wav", std::ios::binary | std::ios::ate);
    std::streamsize size = 16000 * 10;
    wavin.seekg(44, std::ios::beg);

    g_audio_buffer.resize(size);
    wavin.read(reinterpret_cast<char*>(g_audio_buffer.data()), size * sizeof(float));

    for (auto& sample : g_audio_buffer) {
        sample *= 32767.0f;
    }
}

static void BM_ASR_Inference(benchmark::State& state) {
    if (!g_model || g_audio_buffer.empty()) {
        SetupBenchmark(state);
    }
    for (auto _ : state) {    
        benchmark::DoNotOptimize((*g_model)(g_audio_buffer));
    }
}

BENCHMARK(BM_ASR_Inference)
    ->MinWarmUpTime(2.0)
    ->MinTime(5.0)
    ->Repetitions(10)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
