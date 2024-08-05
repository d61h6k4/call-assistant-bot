#include "benchmark/benchmark.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/ocr/model.h"

static void BM_OCR(benchmark::State &state) {

  auto model = aikit::ml::OCR("ml/ocr/models/model.onnx");

  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/participant_name.png"), input_mat,
               cv::COLOR_BGR2GRAY);

  std::string res = model(input_mat.data);

  for (auto _ : state) {
    res = model(input_mat.data);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_OCR)->MinWarmUpTime(2.0)->MinTime(5.0);
BENCHMARK_MAIN();
