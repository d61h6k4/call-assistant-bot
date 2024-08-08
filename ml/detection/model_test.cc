
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/detection/model.h"

#include "absl/log/absl_log.h"


TEST(TestMLDetectionModel, SanityCheck) {
  auto model = aikit::ml::CDetr("ml/detection/models/model.onnx");

  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/meeting_frame.png"), input_mat,
               cv::COLOR_BGR2RGB);
  auto det = model(input_mat.data);

  EXPECT_EQ(det.size(), 10);
  for (auto d : det) {
    ABSL_LOG(INFO) << "[" << d.x_center << ", " << d.y_center << ", " << d.width << ", "
                   << d.height << "] " << d.label_id << " " << d.score << "\n";
  }
}
