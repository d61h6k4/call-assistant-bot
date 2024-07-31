#include "gtest/gtest.h"
#include <memory>

// clang-format off
#include "filesystem.h"
#include "logging.h"
#include "config.h"
#include "models/florence2/model.h"
#include "models/florence2/processor.h"
#include "ort_genai.h"
// clang-format on

TEST(Florence2ProcessorTest, SanityCheck) {
  Generators::SetLogBool("enabled", true);
  auto &log_stream = Generators::Log("Florence2ModelTest::SanityCheck");

  auto config =
      std::make_unique<Generators::Config>(fs::path("models/florence2/data"));
  auto model = std::make_shared<aikit::Florence2>(std::move(config),
                                                  Generators::GetOrtEnv());
  auto tokenizer = model->CreateTokenizer();
  auto processor =
      aikit::Processor(*model->config_.get(), *model->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  EXPECT_EQ(input_tensors->size(), 2)
      << "Size of input tensors is " << input_tensors->size();
}
