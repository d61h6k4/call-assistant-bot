

#include "benchmark/benchmark.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/detection/model.h"

static void BM_CDetr(benchmark::State &state) {
  auto model = aikit::ml::CDetr{};

  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/meeting_frame.png"), input_mat,
               cv::COLOR_BGR2RGB);

  for (auto _ : state) {
    benchmark::DoNotOptimize(model(input_mat.data));
  }
}

BENCHMARK(BM_CDetr)->MinWarmUpTime(2.0)->MinTime(5.0);
BENCHMARK_MAIN();
