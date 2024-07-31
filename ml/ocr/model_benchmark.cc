

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

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


static void BM_CDetr(benchmark::State &state) {

  auto config =
      std::move(std::make_unique<Generators::Config>(fs::path("ml/ocr/models")));
  auto model_ = std::make_shared<aikit::Florence2>(std::move(config),
                                                   Generators::GetOrtEnv());

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  auto tokenizer_stream = tokenizer->CreateStream();

  std::string prompt = "What is the text in the image, with regions?";
  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/participant.png"), input_mat,
               cv::COLOR_BGR2RGB);

  for (auto _ : state) {

    auto input_tensors =
        processor.Process(*tokenizer.get(), prompt, input_mat.data);

    auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
    params->SetInputs(*input_tensors);

    auto generator = Generators::CreateGenerator(*model_.get(), *params);
    // auto token_sequences = Generators::Generate(*model_.get(), *params);

    benchmark::DoNotOptimize(Generators::Generate(*model_.get(), *params));
  }
}

BENCHMARK(BM_CDetr)->MinWarmUpTime(2.0)->MinTime(5.0);
BENCHMARK_MAIN();
