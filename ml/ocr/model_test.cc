
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/ocr/model.h"

#include "absl/log/absl_log.h"

TEST(TestMLOCRModel, SanityCheck) {
  auto model = aikit::ml::OCR("ml/ocr/models/model.onnx");

  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/participant_name.png"), input_mat,
               cv::COLOR_BGR2GRAY);
  auto det = model(input_mat.data);

  ABSL_LOG(INFO) << det;

  det = model(input_mat.data);

  ABSL_LOG(INFO) << det;
}
