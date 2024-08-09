
#include "ml/detection/model.h"
#include <cstdint>
#include <thread>

namespace aikit::ml {
CDetr::CDetr(const std::string& path_to_model) {
  env_ = Ort::Env(logging_level_, log_id_.c_str());

  run_options_ = Ort::RunOptions();
  session_options_ = Ort::SessionOptions();

  // Default to a limit of 16 threads to optimize performance
  constexpr int min_thread_nums = 1;
  constexpr int max_thread_nums = 16;
  int num_of_cores =
      std::max(min_thread_nums,
               static_cast<int>(std::thread::hardware_concurrency() / 2));
  session_options_.SetIntraOpNumThreads(num_of_cores);
  session_options_.SetInterOpNumThreads(num_of_cores);
  session_options_.EnableCpuMemArena();
  session_options_.EnableMemPattern();

  session_options_.SetLogId(log_id_.c_str());
  session_options_.SetLogSeverityLevel(logging_level_);

  // if (options.enable_profiling.has_value()) {
  //   fs::path profile_file_prefix{options.enable_profiling.value()};
  //   ort_options.EnableProfiling(profile_file_prefix.c_str());
  // }

  session_ =
      Ort::Session(env_, path_to_model.c_str(), session_options_);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  allocator_device_ = Ort::Allocator(session_, memory_info);

  input_tensor_ = Ort::Value::CreateTensor<uint8_t>(
      allocator_device_, input_shape_.data(), input_shape_.size());
}

std::vector<Detection> CDetr::operator()(const uint8_t *image) {

  auto input_tensor_data = input_tensor_.GetTensorMutableData<uint8_t>();
  std::copy_n(image, 3 * height * width, input_tensor_data);
  auto output_tensors =
      session_.Run(run_options_, input_names_.data(), &input_tensor_, 1,
                   output_names_.data(), 1);
  auto elements_num =
      output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[0];
  float *floatarr = output_tensors.front().GetTensorMutableData<float>();

  std::vector<Detection> res;
  res.reserve(elements_num);
  for (auto detection_ix = 0; detection_ix < elements_num; ++detection_ix) {
    res.emplace_back(Detection{
        .x_center = floatarr[detection_ix * 6 + 0],
        .y_center = floatarr[detection_ix * 6 + 1],
        .width = floatarr[detection_ix * 6 + 2],
        .height = floatarr[detection_ix * 6 + 3],
        .label_id = static_cast<int>(floatarr[detection_ix * 6 + 5]),
        .score = floatarr[detection_ix * 6 + 4],
    });
  }
  return res;
}

} // namespace aikit::ml
