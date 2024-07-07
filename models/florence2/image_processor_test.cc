
#include "gtest/gtest.h"
#include "ortx_c_helper.h"
#include "shared/api/image_processor.h"

TEST(Florence2ProcessorTest, TestClipImageProcessing) {
  const char *images_path[] = {"models/florence2/data/car.jpg"};
  ort_extensions::OrtxObjectPtr<OrtxRawImages> raw_images;
  extError_t err =
      OrtxLoadImages(ort_extensions::ptr(raw_images), images_path, 1, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(ort_extensions::ptr(processor),
                            "models/florence2/data/processor_config.json");
  ASSERT_EQ(err, kOrtxOK) << "Error: " << OrtxGetLastErrorMessage();

  ort_extensions::OrtxObjectPtr<OrtxImageProcessorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(),
                            ort_extensions::ptr(result));
  ASSERT_EQ(err, kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxImageGetTensorResult(result.get(), 0, ort_extensions::ptr(tensor));
  ASSERT_EQ(err, kOrtxOK);

  const float *data{};
  const int64_t *shape{};
  size_t num_dims;
  err = OrtxGetTensorDataFloat(tensor.get(), &data, &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 4);
}
