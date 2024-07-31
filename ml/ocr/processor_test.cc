#include "gtest/gtest.h"
#include <memory>

// clang-format off
#include "filesystem.h"
#include "logging.h"
#include "config.h"
#include "ml/ocr/model.h"
#include "ml/ocr/processor.h"
#include "ort_genai.h"
// clang-format on

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

TEST(Florence2ProcessorTest, SanityCheck) {
  Generators::SetLogBool("enabled", true);
  auto &log_stream = Generators::Log("Florence2ModelTest::SanityCheck");

  auto config =
      std::make_unique<Generators::Config>(fs::path("ml/ocr/models"));
  auto model = std::make_shared<aikit::Florence2>(std::move(config),
                                                  Generators::GetOrtEnv());
  auto tokenizer = model->CreateTokenizer();
  auto processor =
      aikit::Processor(*model->config_.get(), *model->session_info_.get());

  std::string prompt = "What does the image describe?";
  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/participant.png"), input_mat,
                 cv::COLOR_BGR2RGB);


  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, input_mat.data);

  EXPECT_EQ(input_tensors->size(), 2)
      << "Size of input tensors is " << input_tensors->size();
}
