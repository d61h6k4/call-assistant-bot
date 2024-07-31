
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "models/florence2/model.h"

namespace aikit {

Florence2::Florence2(std::unique_ptr<Generators::Config> config,
                     OrtEnv &ort_env)
    : Generators::Model{std::move(config)} {
  embedding_session_ = OrtSession::Create(
      ort_env,
      (config_->config_path / fs::path(config_->model.embedding.filename))
          .c_str(),
      session_options_.get());

  // User a custom vision session if available; otherwise, fallback to the
  // generic options
  auto *vision_session_options = vision_session_options_
                                     ? vision_session_options_.get()
                                     : session_options_.get();
  // The line loads the customop library into ONNXRuntime engine to load the
  // ONNX model with the custom op. Here we load DecodeImage kernel
  Ort::ThrowOnError(RegisterCustomOps(vision_session_options, OrtGetApiBase()));

  vision_session_ = OrtSession::Create(
      ort_env,
      (config_->config_path / fs::path(config_->model.vision.filename)).c_str(),
      vision_session_options);

  encoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / fs::path("encoder_model.onnx")).c_str(),
      session_options_.get());

  decoder_session_ = OrtSession::Create(
      ort_env,
      (config_->config_path / fs::path(config_->model.decoder.filename))
          .c_str(),
      session_options_.get());

  InitDeviceAllocator(*decoder_session_);
  session_info_->Add(*embedding_session_);
  session_info_->Add(*vision_session_);
  session_info_->Add(*encoder_session_);
}

std::unique_ptr<Generators::State>
Florence2::CreateState(Generators::RoamingArray<int32_t> sequence_lengths,
                       const Generators::GeneratorParams &params) const {
  return std::make_unique<PipelineState>(*this, sequence_lengths, params);
}

PipelineState::PipelineState(
    const Florence2 &model,
    const Generators::RoamingArray<int32_t> &sequence_lengths,
    const Generators::GeneratorParams &params)
    : State{params, model}, model_{model},
      captured_graph_info_{
          model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)},
      embedding_state_{std::make_unique<EmbeddingState>(
          model, params, captured_graph_info_.get())},
      vision_state_{std::make_unique<VisionState>(model_, params)},
      encoder_state_{std::make_unique<EncoderState>(
          model_, params, captured_graph_info_.get())},
      decoder_state_{std::make_unique<DecoderState>(
          model_, sequence_lengths, params, captured_graph_info_.get())} {}

Generators::RoamingArray<float>
PipelineState::Run(int current_length,
                   Generators::RoamingArray<int32_t> next_tokens,
                   Generators::RoamingArray<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  if (is_prompt_) {
    embedding_state_->Run(current_length, next_tokens, next_indices);
    vision_state_->Run(current_length, next_tokens, next_indices);

    encoder_state_->MergeInputIdsWithImageFeatures(
        embedding_state_->inputs_embeds_.get(),
        vision_state_->visual_features_.get());
    encoder_state_->Run(current_length, next_tokens, next_indices);

    decoder_state_->SetUseCacheBranch(false);
    // Expand decoder inputs according the beam size
    decoder_state_->encoder_attention_mask_ = model_.ExpandInputs(
        encoder_state_->attention_mask_, params_->search.num_beams);
    decoder_state_->input_names_.push_back("encoder_attention_mask");
    decoder_state_->inputs_.push_back(
        decoder_state_->encoder_attention_mask_.get());

    decoder_state_->encoder_hidden_states_ = model_.ExpandInputs(
        encoder_state_->last_hidden_state_, params_->search.num_beams);
    decoder_state_->input_names_.push_back("encoder_hidden_states");
    decoder_state_->inputs_.push_back(
        decoder_state_->encoder_hidden_states_.get());

    decoder_state_->inputs_embeds_ = model_.ExpandInputs(
        embedding_state_->inputs_embeds_, params_->search.num_beams);
    decoder_state_->inputs_embeds_index_ = decoder_state_->inputs_.size();
    decoder_state_->input_names_.push_back("inputs_embeds");
    decoder_state_->inputs_.push_back(decoder_state_->inputs_embeds_.get());
    auto logits =
        decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;

    decoder_state_->SetUseCacheBranch(true);
    decoder_state_->encoder_kv_cache_.Update();
    vision_state_.reset();
    // encoder_state_.reset();
    return logits;
  }

  embedding_state_->UpdateInputsAndOutputs(next_tokens);
  decoder_state_->UpdateInputsOutputs(current_length, next_indices);

  embedding_state_->Run(current_length, next_tokens, next_indices);
  decoder_state_->inputs_[decoder_state_->inputs_embeds_index_] =
      embedding_state_->outputs_[embedding_state_->inputs_embeds_index_];
  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

EmbeddingState::EmbeddingState(
    const Florence2 &model, const Generators::GeneratorParams &params,
    const Generators::CapturedGraphInfo *captured_graph_info)
    : State{params, model}, model_{model},
      captured_graph_info_{captured_graph_info} {

  // Create input int tensor
  input_ids_name_ = "input_ids";
  if (!model_.session_info_->HasInput(input_ids_name_)) {
    throw std::runtime_error(input_ids_name_ +
                             " not found in the model input.");
  }
  input_ids_type_ = model_.session_info_->GetInputDataType(input_ids_name_);
  // sequence length here is equal to number of tokens in prompt
  // Set in GeneratorParams::SetInputs
  // Create and fill tensor (GeneratorParams::SetInputs contains input_ids)
  input_ids_shape_ = {params_->batch_size, params_->sequence_length};
  if (input_ids_type_ == Ort::TypeToTensorType<int32_t>::type) {
    input_ids_ = OrtValue::CreateTensor<int32_t>(
        model_.allocator_cpu_.GetInfo(),
        Generators::cpu_span<int32_t>(
            const_cast<int32_t *>(params_->input_ids.data()),
            input_ids_shape_[0] * input_ids_shape_[1]),
        input_ids_shape_);
  } else if (input_ids_type_ == Ort::TypeToTensorType<int64_t>::type) {
    // If 64-bit, convert from 32-bit to 64-bit
    input_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_, input_ids_shape_,
                                        input_ids_type_);
    auto *p_data = input_ids_->GetTensorMutableData<int64_t>();
    for (auto v : params_->input_ids) {
      *p_data++ = v;
    }
  } else {
    throw std::runtime_error("Unsupported data type for " + input_ids_name_ +
                             ": " + std::to_string(input_ids_type_));
  }
  input_ids_index_ = inputs_.size();
  inputs_.push_back(input_ids_.get());
  input_names_.push_back(input_ids_name_.c_str());

  // Create inputs_embeds
  inputs_embeds_name_ = "inputs_embeds";
  inputs_embeds_type_ =
      model_.session_info_->GetOutputDataType(inputs_embeds_name_);
  inputs_embeds_shape_ = {params_->batch_size, params_->sequence_length,
                          params_->hidden_size};
  inputs_embeds_ = OrtValue::CreateTensor(
      *model_.allocator_device_, inputs_embeds_shape_, inputs_embeds_type_);
  inputs_embeds_index_ = outputs_.size();
  outputs_.push_back(inputs_embeds_.get());
  output_names_.push_back(inputs_embeds_name_.c_str());
}

void EmbeddingState::UpdateInputsAndOutputs(
    Generators::RoamingArray<int32_t> next_tokens_unk) {
  // input_ids_.Update(next_tokens);
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (input_ids_shape_[1] != 1) {
    input_ids_shape_[0] = params_->BatchBeamSize();
    input_ids_shape_[1] = 1;
    input_ids_ = OrtValue::CreateTensor(*model_.allocator_device_,
                                        input_ids_shape_, input_ids_type_);
    inputs_[input_ids_index_] = input_ids_.get();
  }
  // Update input_ids with next tokens, converting from 32-bit to 64-bit
  if (input_ids_type_ == Ort::TypeToTensorType<int64_t>::type) {
    auto *data = input_ids_->GetTensorMutableData<int64_t>();
    auto next_tokens = next_tokens_unk.GetCPU();
    for (int i = 0; i < input_ids_shape_[0]; i++) {
      data[i] = next_tokens[i];
    }
  } else {
    auto *data = input_ids_->GetTensorMutableData<int32_t>();
    memcpy(data, next_tokens_unk.GetCPU().data(),
           input_ids_shape_[0] * sizeof(int32_t));
  }

  // inputs_embeds_.UpdateSequenceLength();
  if (inputs_embeds_shape_[1] != 1) {
    inputs_embeds_shape_[0] = params_->BatchBeamSize();
    inputs_embeds_shape_[1] = 1;
    inputs_embeds_ = OrtValue::CreateTensor(
        *model_.allocator_device_, inputs_embeds_shape_, inputs_embeds_type_);
    outputs_[inputs_embeds_index_] = inputs_embeds_.get();
  }
}

Generators::RoamingArray<float>
EmbeddingState::Run(int current_length,
                    Generators::RoamingArray<int32_t> next_tokens,
                    Generators::RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_shape_[0]);
  State::Run(*model_.embedding_session_, *model_.run_options_, batch_size);

  return Generators::RoamingArray<float>();
}

VisionState::VisionState(const Florence2 &model,
                         const Generators::GeneratorParams &params)
    : State{params, model}, model_{model} {

  // Create pixel_values tensor
  // pixel_values tensor is already created by processor
  // so here we simply register it as an inptut
  auto [pixel_values_name, pixel_values_in_graph] =
      model_.config_->GetGraphName(
          std::string(Generators::Config::Defaults::PixelValuesName));
  if (!pixel_values_in_graph) {
    throw std::runtime_error(
        "Could not find " +
        std::string(Generators::Config::Defaults::PixelValuesName) +
        " in extra inputs. Please create tensor with it (you can use processor "
        "for that) and use GeneratorParams::SetInput to register it.");
  }
  for (auto &extra_input : params_->extra_inputs) {
    if (extra_input.name == pixel_values_name) {
      input_names_.push_back(extra_input.name.c_str());
      inputs_.push_back(extra_input.tensor->ort_tensor_.get());
    }
  }

  // Create visual features

  constexpr int32_t batch_size = 1;
  if (params_->batch_size != batch_size) {
    throw std::runtime_error(
        "Vision encoder supports only batch_size=1 scenario.");
  }
  visual_features_name_ = model_.config_->model.vision.outputs.visual_features;
  if (!model_.session_info_->HasOutput(visual_features_name_)) {
    throw std::runtime_error("Visual features output not found in the model");
  }

  auto visual_features_type =
      model_.session_info_->GetOutputDataType(visual_features_name_);
  std::vector<int64_t> shape = {batch_size, VisionState::num_image_tokens,
                                params_->hidden_size};

  switch (visual_features_type) {
  case Ort::TypeToTensorType<float>::type:
    visual_features_ =
        OrtValue::CreateTensor<float>(*model_.allocator_device_, shape);
    break;
  case Ort::TypeToTensorType<Ort::Float16_t>::type:
    visual_features_ = OrtValue::CreateTensor<Ort::Float16_t>(
        *model_.allocator_device_, shape);
    break;
  default:
    throw std::runtime_error("Unsupported data type for visual features: " +
                             std::to_string(visual_features_type));
  }
  output_names_.push_back(visual_features_name_.c_str());
  outputs_.push_back(visual_features_.get());
}

Generators::RoamingArray<float>
VisionState::Run(int current_length,
                 Generators::RoamingArray<int32_t> next_tokens,
                 Generators::RoamingArray<int32_t> next_indices) {
  State::Run(*model_.vision_session_, *model_.run_options_, 1);

  return Generators::RoamingArray<float>();
}

EncoderState::EncoderState(
    const Florence2 &model, const Generators::GeneratorParams &params,
    const Generators::CapturedGraphInfo *captured_graph_info)
    : State{params, model}, model_{model},
      captured_graph_info_{captured_graph_info} {

  auto inputs_embeds_type =
      model_.session_info_->GetInputDataType("inputs_embeds");

  std::vector<int64_t> inputs_embeds_shape = {params_->batch_size,
                                              params_->sequence_length +
                                                  VisionState::num_image_tokens,
                                              params_->hidden_size};

  switch (inputs_embeds_type) {
  case Ort::TypeToTensorType<float>::type:
    inputs_embeds_ = OrtValue::CreateTensor<float>(*model_.allocator_device_,
                                                   inputs_embeds_shape);
    break;
  case Ort::TypeToTensorType<Ort::Float16_t>::type:
    inputs_embeds_ = OrtValue::CreateTensor<Ort::Float16_t>(
        *model_.allocator_device_, inputs_embeds_shape);
    break;
  default:
    throw std::runtime_error(
        "Unsupported data type for encoder inputs embeds: " +
        std::to_string(inputs_embeds_type));
  }
  input_names_.push_back("inputs_embeds");
  inputs_.push_back(inputs_embeds_.get());

  auto attention_mask_type =
      model_.session_info_->GetInputDataType("attention_mask");

  std::vector<int64_t> attention_mask_shape = {
      params_->batch_size,
      params_->sequence_length + VisionState::num_image_tokens};

  switch (attention_mask_type) {
  case Ort::TypeToTensorType<int64_t>::type:
    attention_mask_ = OrtValue::CreateTensor<int64_t>(*model_.allocator_device_,
                                                      attention_mask_shape);
    break;
  case Ort::TypeToTensorType<int32_t>::type:
    attention_mask_ = OrtValue::CreateTensor<int32_t>(*model_.allocator_device_,
                                                      attention_mask_shape);
    break;
  default:
    throw std::runtime_error(
        "Unsupported data type for encoder attention mask: " +
        std::to_string(attention_mask_type));
  }
  attention_mask_index_ = inputs_.size();
  input_names_.push_back("attention_mask");
  inputs_.push_back(attention_mask_.get());

  auto last_hidden_state_type =
      model_.session_info_->GetOutputDataType("last_hidden_state");

  std::vector<int64_t> last_hidden_state_shape = {
      params_->batch_size,
      params_->sequence_length + VisionState::num_image_tokens,
      params_->hidden_size};

  switch (last_hidden_state_type) {
  case Ort::TypeToTensorType<float>::type:
    last_hidden_state_ = OrtValue::CreateTensor<float>(
        *model_.allocator_device_, last_hidden_state_shape);
    break;
  case Ort::TypeToTensorType<Ort::Float16_t>::type:
    last_hidden_state_ = OrtValue::CreateTensor<Ort::Float16_t>(
        *model_.allocator_device_, last_hidden_state_shape);
    break;
  default:
    throw std::runtime_error(
        "Unsupported data type for encoder last hidden state: " +
        std::to_string(last_hidden_state_type));
  }
  last_hidden_state_index_ = outputs_.size();
  output_names_.push_back("last_hidden_state");
  outputs_.push_back(last_hidden_state_.get());
}

void EncoderState::MergeInputIdsWithImageFeatures(
    const OrtValue *inputs_embeds, const OrtValue *image_features) {

  const int64_t sequence_length = params_->sequence_length;
  const int64_t hidden_size = params_->hidden_size;
  // Encoder's inputs_embeds
  auto target = Generators::cpu_span<float>(
      inputs_embeds_->GetTensorMutableData<float>(),
      (VisionState::num_image_tokens + sequence_length) * hidden_size);
  auto image_target =
      target.subspan(0, VisionState::num_image_tokens * hidden_size);
  auto image_source = Generators::cpu_span<const float>(
      image_features->GetTensorData<float>(),
      VisionState::num_image_tokens * hidden_size);
  std::copy(image_source.begin(), image_source.end(), image_target.begin());
  auto prompt_target =
      target.subspan(VisionState::num_image_tokens * hidden_size,
                     sequence_length * hidden_size);
  auto prompt_source = Generators::cpu_span<const float>(
      inputs_embeds->GetTensorData<float>(), sequence_length * hidden_size);
  std::copy(prompt_source.begin(), prompt_source.end(), prompt_target.begin());

  auto attention_mask = Generators::cpu_span<int64_t>(
      attention_mask_->GetTensorMutableData<int64_t>(),
      VisionState::num_image_tokens + sequence_length);
  std::fill(attention_mask.begin(), attention_mask.end(), 1);
}

Generators::RoamingArray<float>
EncoderState::Run(int current_length,
                  Generators::RoamingArray<int32_t> next_tokens,
                  Generators::RoamingArray<int32_t> next_indices) {
  State::Run(*model_.encoder_session_, *model_.run_options_, 1);
  return Generators::RoamingArray<float>();
}

DecoderState::DecoderState(
    const Florence2 &model, Generators::RoamingArray<int32_t> sequence_lengths,
    const Generators::GeneratorParams &params,
    const Generators::CapturedGraphInfo *captured_graph_info)
    : State{params, model}, model_{model},
      captured_graph_info_{captured_graph_info} {

  logits_.Add();
  kv_cache_.Add();
  encoder_kv_cache_.AddEncoder();

  auto use_cache_branch_type =
      model_.session_info_->GetInputDataType("use_cache_branch");

  std::vector<int64_t> use_cache_branch_shape = {1};
  use_cache_branch_ = OrtValue::CreateTensor<bool>(*model_.allocator_device_,
                                                   use_cache_branch_shape);
  input_names_.push_back("use_cache_branch");
  inputs_.push_back(use_cache_branch_.get());

  // encoder attention mask is created in Run from encoder's attention mask
  // encoder hidden states is created in Run
}

Generators::RoamingArray<float>
DecoderState::Run(int current_length,
                  Generators::RoamingArray<int32_t> next_tokens,
                  Generators::RoamingArray<int32_t> next_indices) {
  State::Run(*model_.decoder_session_, *model_.run_options_, 1);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(
    int current_length, Generators::RoamingArray<int32_t> beam_indices) {
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  logits_.Update();
}

void DecoderState::SetUseCacheBranch(bool use_cache_branch) {
  use_cache_branch_->GetTensorMutableData<bool>()[0] = use_cache_branch;
}
} // namespace aikit
