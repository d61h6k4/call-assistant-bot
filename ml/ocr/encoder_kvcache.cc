#include "models/model.h"

#include "models/florence2/encoder_kvcache.h"

namespace Generators {

EncoderKVCache::EncoderKVCache(const Model &model, State &state)
    : model_{model}, state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{
          state_.params_->search.past_present_share_buffer &&
          state_.params_->search.num_beams == 1},
      shape_{state_.params_->BatchBeamSize(),
             model.config_->model.decoder.num_key_value_heads, 0,
             model.config_->model.decoder.head_size} {

  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    char string[64];
    snprintf(string, std::size(string),
             model.config_->model.decoder.inputs.cross_past_key_names.c_str(),
             i);
    input_name_strings_.emplace_back(string);
    snprintf(string, std::size(string),
             model.config_->model.decoder.inputs.cross_past_value_names.c_str(),
             i);
    input_name_strings_.emplace_back(string);

    snprintf(
        string, std::size(string),
        model.config_->model.decoder.outputs.cross_present_key_names.c_str(),
        i);
    output_name_strings_.emplace_back(string);
    snprintf(
        string, std::size(string),
        model.config_->model.decoder.outputs.cross_present_value_names.c_str(),
        i);
    output_name_strings_.emplace_back(string);
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  empty_past_ =
      OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

  shape_[2] = state_.params_->sequence_length + 577;

  for (int i = 0; i < layer_count_ * 2; ++i) {
    presents_.push_back(
        OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_));
  }
}

void EncoderKVCache::AddEncoder() {
  output_index_ = state_.outputs_.size();

  // We don't set the input_index_ & output_index_ because the encoder step only
  // runs once, there's no update
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void EncoderKVCache::Update() {
  input_index_ = state_.inputs_.size();
  for (int i = 0; i < layer_count_ * 2; i++) {
    pasts_[i] = std::move(presents_[i]);
    state_.inputs_.push_back(pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
  }
  shape_[0] = 0;
  shape_[2] = 1;
  for (int i = 0; i < layer_count_ * 2; i++) {
    presents_[i] =
        OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

} // namespace Generators
