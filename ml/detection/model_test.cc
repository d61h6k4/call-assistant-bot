
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ml/detection/model.h"

#include "absl/log/absl_log.h"

namespace {
cv::Mat hwc2chw(const cv::Mat &image) {
  std::vector<cv::Mat> rgb_images;
  cv::split(image, rgb_images);

  // Stretch one-channel images to vector
  cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
  cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
  cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

  // Now we can rearrange channels if need
  cv::Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};

  cv::Mat flat_image;
  // Concatenate three vectors to one
  cv::hconcat(matArray, 3, flat_image);
  return flat_image;
}
} // namespace

TEST(TestMLDetectionModel, SanityCheck) {
  auto model = aikit::ml::CDetr{};

  cv::Mat input_mat;
  cv::cvtColor(cv::imread("testdata/meeting_frame.png"), input_mat,
               cv::COLOR_BGR2RGB);
  auto chw_mat = hwc2chw(input_mat);
  auto det = model(chw_mat.data);

  EXPECT_EQ(det.size(), 5);
  for (auto d : det) {
    ABSL_LOG(INFO) << "[" << d.xmin << ", " << d.ymin << ", " << d.width << ", "
                   << d.height << "] " << d.label_id << " " << d.score << "\n";
  }
}