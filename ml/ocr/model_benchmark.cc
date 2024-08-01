

#include "benchmark/benchmark.h"

// clang-format off
#include "generators.h"
#include "filesystem.h"
#include "logging.h"
#include "config.h"
#include "ml/ocr/model.h"
#include "ml/ocr/processor.h"
#include "search.h"
// clang-format on

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class Florence2Fixture : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State &) override {

    auto config = std::move(
        std::make_unique<Generators::Config>(fs::path("ml/ocr/models")));
    model = std::make_shared<aikit::Florence2>(std::move(config),
                                               Generators::GetOrtEnv());

    tokenizer = model->CreateTokenizer();
    processor = std::make_shared<aikit::Processor>(*model->config_.get(),
                                                   *model->session_info_.get());

    tokenizer_stream = tokenizer->CreateStream();

    cv::cvtColor(cv::imread("testdata/participant.png"), input_mat,
                 cv::COLOR_BGR2RGB);
  }

protected:
  std::shared_ptr<aikit::Florence2> model;
  std::shared_ptr<Generators::Tokenizer> tokenizer;
  std::shared_ptr<aikit::Processor> processor;
  std::shared_ptr<Generators::TokenizerStream> tokenizer_stream;

  std::string prompt = "What is the text in the image, with regions?";
  cv::Mat input_mat;
};

BENCHMARK_F(Florence2Fixture, BM_Florence2e2e)(benchmark::State &state) {
  for (auto _ : state) {

    auto input_tensors =
        processor->Process(*tokenizer.get(), prompt, input_mat.data);

    auto params = std::make_shared<Generators::GeneratorParams>(*model.get());
    params->SetInputs(*input_tensors);

    benchmark::DoNotOptimize(Generators::Generate(*model.get(), *params));
  }
}

BENCHMARK_F(Florence2Fixture, BM_Florence2processor)(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        processor->Process(*tokenizer.get(), prompt, input_mat.data));
  }
}

BENCHMARK_F(Florence2Fixture, BM_Florence2setInputs)(benchmark::State &state) {
  for (auto _ : state) {

    auto input_tensors =
        processor->Process(*tokenizer.get(), prompt, input_mat.data);

    auto params = std::make_shared<Generators::GeneratorParams>(*model.get());
    params->SetInputs(*input_tensors);
  }
}

BENCHMARK_F(Florence2Fixture, BM_Florence2encoder)(benchmark::State &state) {
  for (auto _ : state) {

    auto input_tensors =
        processor->Process(*tokenizer.get(), prompt, input_mat.data);

    auto params = std::make_shared<Generators::GeneratorParams>(*model.get());
    params->SetInputs(*input_tensors);

    auto generator = Generators::CreateGenerator(*model.get(), *params);
    generator->ComputeLogits();
  }
}

BENCHMARK_REGISTER_F(Florence2Fixture, BM_Florence2encoder)
    ->MinWarmUpTime(2.0)
    ->MinTime(5.0);

BENCHMARK_REGISTER_F(Florence2Fixture, BM_Florence2e2e)
    ->MinWarmUpTime(2.0)
    ->MinTime(5.0);
BENCHMARK_MAIN();
