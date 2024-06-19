
#include "absl/log/absl_log.h"
#include "av_transducer/utils/video.h"
#include "gtest/gtest.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/imgutils.h"
#ifdef __cplusplus
}
#endif

std::vector<uint8_t> GenerateImage(int width, int height, int frame_ix) {
  auto pix_fmt = AV_PIX_FMT_YUV420P;
  auto size = av_image_get_buffer_size(pix_fmt, width, height, 1);
  std::vector<uint8_t> image(size);
  std::array<int, 4> linesize{};
  auto res = av_image_fill_linesizes(linesize.data(), pix_fmt, width);
  EXPECT_GE(res, 0);
  /* prepare a dummy image */
  /* Y */
  for (auto y = 0; y < height; y++) {
    for (auto x = 0; x < width; x++) {
      image[y * linesize[0] + x] = x + y + frame_ix * 3;
    }
  }

  /* Cb and Cr */
  for (auto y = 0; y < height / 2; y++) {
    for (auto x = 0; x < width / 2; x++) {
      image[width * height + y * linesize[1] + x] = 128 + y + frame_ix * 2;
      image[width * height + width / 2 * linesize[1] + y * linesize[2] + x] =
          64 + x + frame_ix * 5;
    }
  }
  return image;
}

TEST(TestVideoUtils, CheckCreateFrame) {

  auto video_frame =
      aikit::media::VideoFrame::CreateVideoFrame(AV_PIX_FMT_YUV420P, 1280, 720);
  EXPECT_TRUE(video_frame->c_frame());
}

TEST(TestVideoUtils, CheckWriteReadFrame) {
  auto video_frame =
      aikit::media::VideoFrame::CreateVideoFrame(AV_PIX_FMT_YUV420P, 3, 3);
  auto image = GenerateImage(3, 3, 1);
  auto st = video_frame->CopyFromBuffer(image.data());
  EXPECT_TRUE(st.ok());

  auto image_copy = std::vector<uint8_t>(image.size());
  st = video_frame->CopyToBuffer(image_copy.data());
  EXPECT_TRUE(st.ok());

  for (auto ix = 0; ix < image.size(); ++ix) {
     EXPECT_EQ(image[ix], image_copy[ix]);
  }
}
