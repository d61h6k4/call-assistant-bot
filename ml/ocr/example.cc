#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
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

bool FileExists(const char *path) {
  return static_cast<bool>(std::ifstream(path));
}

void CXX_API() {
  std::cout << "Creating model..." << std::endl;

  auto config = std::move(
      std::make_unique<Generators::Config>(fs::path("ml/ocr/models")));
  auto model_ = std::make_shared<aikit::Florence2>(std::move(config),
                                                   Generators::GetOrtEnv());
  std::cout << "Creating multimodal processor..." << std::endl;

  auto tokenizer = model_->CreateTokenizer();
  auto processor =
      aikit::Processor(*model_->config_.get(), *model_->session_info_.get());

  auto tokenizer_stream = tokenizer->CreateStream();

  std::string prompt = "What is the text in the image, with regions?";
  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/participant.png"), input_mat,
               cv::COLOR_BGR2RGB);

  std::cout << "Processing image and prompt..." << std::endl;
  auto input_tensors =
      processor.Process(*tokenizer.get(), prompt, input_mat.data);

  std::cout << "Generating response..." << std::endl;
  auto params = std::make_shared<Generators::GeneratorParams>(*model_.get());
  params->SetInputs(*input_tensors);

  auto generator = Generators::CreateGenerator(*model_.get(), *params);
  auto token_sequences = Generators::Generate(*model_.get(), *params);

  for (auto &token_sequence : token_sequences) {
    std::cout << "Token sequence:\n";
    for (auto token_id : token_sequence) {
      std::cout << tokenizer_stream->Decode(token_id);
    }
    std::cout << "\nLast token id is: " << token_sequence[0] << std::endl;
  }
}

static void print_usage(int /*argc*/, char **argv) {
  std::cerr << "usage: " << argv[0] << " <model_path>" << std::endl;
}

int main(int argc, char **argv) {

  std::cout << "---------------" << std::endl;
  std::cout << "Hello, Falcon2!" << std::endl;
  std::cout << "---------------" << std::endl;

  std::cout << "C++ API" << std::endl;
  CXX_API();

  return 0;
}
