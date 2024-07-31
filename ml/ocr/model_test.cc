
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <memory>

// clang-format off
#include "filesystem.h"
#include "logging.h"
#include "config.h"
#include "models/florence2/model.h"
#include "models/florence2/processor.h"
#include "search.h"
// clang-format on

class Florence2Test : public testing::Test {
protected:
  Florence2Test() {
    Generators::SetLogBool("enabled", true);
    Generators::SetLogBool("model_input_values", true);
    Generators::SetLogBool("model_output_shapes", true);
    Generators::SetLogBool("model_output_values", true);
    Generators::SetLogBool("generate_next_token", true);
    Generators::SetLogBool("model_logits", true);
    Generators::SetLogBool("hit_eos", true);

    auto config = std::move(std::make_unique<Generators::Config>(
        fs::path("models/florence2/data")));
    model_ = std::make_shared<aikit::Florence2>(std::move(config),
                                                Generators::GetOrtEnv());
  }

  // std::unique_ptr<Generators::Config> config_;
  std::shared_ptr<aikit::Florence2> model_;
};

TEST_F(Florence2Test, SanityCheckEmbedding) {

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto captured_graph_info =
      model_->GetCapturedGraphPool()->ReserveCapturedGraph(*model_.get(),
                                                           *params);
  auto emb_state =
      aikit::EmbeddingState(*model_.get(), *params, captured_graph_info.get());
  emb_state.Run(0, {}, {});
}

TEST_F(Florence2Test, SanityCheckVision) {

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto captured_graph_info =
      model_->GetCapturedGraphPool()->ReserveCapturedGraph(*model_.get(),
                                                           *params);

  auto vision_state = aikit::VisionState(*model_.get(), *params);
  vision_state.Run(0, {}, {});
}

TEST_F(Florence2Test, SanityCheckEncoder) {

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto captured_graph_info =
      model_->GetCapturedGraphPool()->ReserveCapturedGraph(*model_.get(),
                                                           *params);

  auto emb_state =
      aikit::EmbeddingState(*model_.get(), *params, captured_graph_info.get());
  auto vision_state = aikit::VisionState(*model_.get(), *params);
  auto encoder_state =
      aikit::EncoderState(*model_.get(), *params, captured_graph_info.get());

  emb_state.Run(0, {}, {});
  vision_state.Run(0, {}, {});

  encoder_state.MergeInputIdsWithImageFeatures(emb_state.outputs_[0],
                                               vision_state.outputs_[0]);
  encoder_state.Run(0, {}, {});
}

TEST_F(Florence2Test, SanityCheckPipeline) {

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto captured_graph_info =
      model_->GetCapturedGraphPool()->ReserveCapturedGraph(*model_.get(),
                                                           *params);
  auto pipeline_state = model_->CreateState({}, *params);
  pipeline_state->Run(0, {}, {});
  std::array<int32_t, 3> next_tokens_unk = {0, 1, 3};
  Generators::cpu_span<int32_t> next_tokens{next_tokens_unk};
  std::array<int32_t, 3> next_indices_unk = {1, 1, 1};
  Generators::cpu_span<int32_t> next_indices{next_indices_unk};
  pipeline_state->Run(9, next_tokens, next_indices);
}

TEST_F(Florence2Test, SanityCheckGenerate) {

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  std::string prompt = "What does the image describe?";
  auto images = Generators::LoadImageImpl("models/florence2/data/car.png");
  EXPECT_EQ(images->num_images_, 1);

  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, images.get());

  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto generator = Generators::CreateGenerator(*model_.get(), *params);
  EXPECT_FALSE(generator->IsDone());
  generator->ComputeLogits();
  generator->GenerateNextToken();
  if (!generator->IsDone()) {
    generator->ComputeLogits();
  }
}
