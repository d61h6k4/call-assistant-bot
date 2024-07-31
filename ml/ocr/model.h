
#pragma once

#include <cstddef>
#include <memory>

// clang-format off
#include "models/model.h"
#include "models/embeddings.h"
#include "models/extra_inputs.h"
#include "models/input_ids.h"
#include "models/position_inputs.h"
#include "models/kv_cache.h"
#include "models/logits.h"
// clang-format on
#include "ml/ocr/encoder_kvcache.h"

namespace aikit {

class Florence2 : public Generators::Model {

public:
  Florence2(std::unique_ptr<Generators::Config> config, OrtEnv &ort_env);

  Florence2(Florence2 &&) = default;
  Florence2 &operator=(Florence2 &&) = default;
  Florence2(const Florence2 &) = delete;
  Florence2 &operator=(const Florence2 &) = delete;

  ~Florence2() final = default;

  std::unique_ptr<Generators::State>
  CreateState(Generators::RoamingArray<int32_t> sequence_lengths,
              const Generators::GeneratorParams &params) const override;

public:
  std::unique_ptr<OrtSession> embedding_session_; // input_ids -> inputs_embeds
  std::unique_ptr<OrtSession> vision_session_; // pixel_values -> image_features
  std::unique_ptr<OrtSession>
      encoder_session_; // [image_features,inputs_embeds] -> last_hidden_state
  std::unique_ptr<OrtSession> decoder_session_;
};

class EmbeddingState : public Generators::State {
public:
  EmbeddingState(const Florence2 &model,
                 const Generators::GeneratorParams &params,
                 const Generators::CapturedGraphInfo *captured_graph_info);
  EmbeddingState(const EmbeddingState &) = delete;
  EmbeddingState &operator=(const EmbeddingState &) = delete;

  Generators::RoamingArray<float>
  Run(int current_length, Generators::RoamingArray<int32_t> next_tokens,
      Generators::RoamingArray<int32_t> next_indices = {}) override;

  const Generators::CapturedGraphInfo *GetCapturedGraphInfo() const override {
    return captured_graph_info_;
  };

private:
  friend class PipelineState;

  void UpdateInputsAndOutputs(Generators::RoamingArray<int32_t> next_tokens);

  const Florence2 &model_;
  const Generators::CapturedGraphInfo *captured_graph_info_;

  // Model input is sequence of tokens
  // We can't use InputIDs here because we do not want to expand
  // this inputs. We are going to expand after merging with image
  // features
  // Shape: [batch_size, sequence_length]
  std::unique_ptr<OrtValue> input_ids_;
  std::vector<int64_t> input_ids_shape_;
  ONNXTensorElementDataType input_ids_type_;
  std::string input_ids_name_;
  size_t input_ids_index_;
  // Output is embeddings of tokens
  // Shape: [batch_size, sequence_length, hidden_value]
  std::unique_ptr<OrtValue> inputs_embeds_;
  std::vector<int64_t> inputs_embeds_shape_;
  ONNXTensorElementDataType inputs_embeds_type_;
  std::string inputs_embeds_name_;
  size_t inputs_embeds_index_;
};

class VisionState : public Generators::State {
public:
  // preprocessor_config.json::image_seq_length
  static constexpr int32_t num_image_tokens = 577;

public:
  VisionState(const Florence2 &model,
              const Generators::GeneratorParams &params);
  VisionState(const VisionState &) = delete;
  VisionState &operator=(const VisionState &) = delete;

  Generators::RoamingArray<float>
  Run(int current_length, Generators::RoamingArray<int32_t> next_tokens,
      Generators::RoamingArray<int32_t> next_indices = {}) override;

private:
  friend class PipelineState;

  const Florence2 &model_;

  // Input of the vision encoder
  std::unique_ptr<OrtValue> pixel_values_;

  // Output of the vision encoder
  std::unique_ptr<OrtValue> visual_features_;
  std::string visual_features_name_;
};

class EncoderState : public Generators::State {
public:
  EncoderState(const Florence2 &model,
               const Generators::GeneratorParams &params,
               const Generators::CapturedGraphInfo *captured_graph_info);
  EncoderState(const EncoderState &) = delete;
  EncoderState &operator=(const EncoderState &) = delete;

  Generators::RoamingArray<float>
  Run(int current_length, Generators::RoamingArray<int32_t> next_tokens,
      Generators::RoamingArray<int32_t> next_indices = {}) override;

  // Initialize inputs_embeds_ with given input ids embeddings and image
  // features and initialize attention mask accordingly
  void MergeInputIdsWithImageFeatures(const OrtValue *inputs_embeds,
                                      const OrtValue *image_features);

private:
  friend class PipelineState;

  void UpdateInputsOutputs(int current_length,
                           Generators::RoamingArray<int32_t> beam_indices);

  const Florence2 &model_;
  const Generators::CapturedGraphInfo *captured_graph_info_;
  // Input of the Encoder. Concatenation of the image and prompt features.
  // Shape: [batch_size=1, image_tokens + prompt_tokens, hidden_size]
  std::unique_ptr<OrtValue> inputs_embeds_;
  std::unique_ptr<OrtValue> attention_mask_;
  // To reuse
  size_t attention_mask_index_;
  // Output of the encoder.
  std::unique_ptr<OrtValue> last_hidden_state_;
  size_t last_hidden_state_index_;
};

struct DecoderState : Generators::State {
  DecoderState(const Florence2 &model,
               Generators::RoamingArray<int32_t> sequence_lengths,
               const Generators::GeneratorParams &params,
               const Generators::CapturedGraphInfo *captured_graph_info);
  DecoderState(const DecoderState &) = delete;
  DecoderState &operator=(const DecoderState &) = delete;

  Generators::RoamingArray<float>
  Run(int current_length, Generators::RoamingArray<int32_t> next_tokens,
      Generators::RoamingArray<int32_t> next_indices) override;

  const Generators::CapturedGraphInfo *GetCapturedGraphInfo() const override {
    return captured_graph_info_;
  };

  void SetUseCacheBranch(bool use_cache_branch);

private:
  friend class PipelineState;

  void UpdateInputsOutputs(int current_length,
                           Generators::RoamingArray<int32_t> beam_indices);

  const Florence2 &model_;
  const Generators::CapturedGraphInfo *captured_graph_info_;
  std::unique_ptr<OrtValue> inputs_embeds_;
  size_t inputs_embeds_index_;
  Generators::KV_Cache kv_cache_{model_, *this};     // Model input
  Generators::EncoderKVCache encoder_kv_cache_{model_, *this};
  std::unique_ptr<OrtValue> use_cache_branch_;       // Model input
  std::unique_ptr<OrtValue> encoder_hidden_states_;  // Model input
  size_t encoder_hidden_states_index_;
  std::unique_ptr<OrtValue> encoder_attention_mask_; // Model input
  size_t encoder_attention_mask_index_;
  Generators::Logits logits_{model_, *this};         // Model output
};

class PipelineState : public Generators::State {

public:
  PipelineState(const Florence2 &model,
                const Generators::RoamingArray<int32_t> &sequence_lengths,
                const Generators::GeneratorParams &params);
  PipelineState &operator=(const PipelineState &) = delete;
  PipelineState(const PipelineState &) = delete;

  Generators::RoamingArray<float>
  Run(int current_length, Generators::RoamingArray<int32_t> next_tokens,
      Generators::RoamingArray<int32_t> next_indices) override;

private:
  void UpdateInputsOutputs(const Generators::RoamingArray<int32_t> &next_tokens,
                           Generators::RoamingArray<int32_t> next_indices,
                           int current_length);

  const Florence2 &model_;
  const Generators::CapturedGraphInfoPtr captured_graph_info_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<EncoderState> encoder_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  bool is_prompt_{true};
};
} // namespace aikit
