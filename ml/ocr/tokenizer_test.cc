
#include "absl/log/absl_log.h"
#include "shared/api/tokenizer_impl.h"
#include "gtest/gtest.h"
#include <memory>

TEST(Florence2TokenizerTest, TestBartTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("ml/ocr/models");
  EXPECT_TRUE(status.IsOk()) << status.ToString();
  EXPECT_NE(tokenizer, nullptr);

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {0,  2264, 16,   5,   2788,
                                              11, 5,    2274, 116, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_1 = {0,    2264, 16, 5,    2788, 11, 5,
                                              2274, 6,    19, 3806, 116,  2};
  std::vector<extTokenId_t> EXPECTED_IDS_2 = {0,    2264, 473, 5,
                                              2274, 6190, 116, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_3 = {
      0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_4 = {
      0, 47066, 21700, 19, 10, 17818, 99, 16, 2343, 11, 5, 2274, 4, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_5 = {
      0, 574, 22486, 5, 8720, 19, 4120, 766, 11, 5, 2274, 4, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_6 = {
      0, 574, 22486, 5, 8720, 11, 5, 2274, 6, 19, 49, 24173, 4, 2};
  std::vector<extTokenId_t> EXPECTED_IDS_7 = {0,  574, 22486, 5, 976, 5327,
                                              11, 5,   2274,  4, 2};
  std::vector<std::string_view> input = {
      "What is the text in the image?",
      "What is the text in the image, with regions?",
      "What does the image describe?",
      "Describe in detail what is shown in the image.",
      "Describe with a paragraph what is shown in the image.",
      "Locate the objects with category name in the image.",
      "Locate the objects in the image, with their descriptions.",
      "Locate the region proposals in the image."};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());

  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
  EXPECT_EQ(token_ids[1], EXPECTED_IDS_1);
  EXPECT_EQ(token_ids[2], EXPECTED_IDS_2);
  EXPECT_EQ(token_ids[3], EXPECTED_IDS_3);
  EXPECT_EQ(token_ids[4], EXPECTED_IDS_4);
  EXPECT_EQ(token_ids[5], EXPECTED_IDS_5);
  EXPECT_EQ(token_ids[6], EXPECTED_IDS_6);
  EXPECT_EQ(token_ids[7], EXPECTED_IDS_7);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {
      EXPECTED_IDS_0, EXPECTED_IDS_1, EXPECTED_IDS_2, EXPECTED_IDS_3,
      EXPECTED_IDS_4, EXPECTED_IDS_5, EXPECTED_IDS_6, EXPECTED_IDS_7};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
  EXPECT_EQ(out_text[1], input[1]);
  EXPECT_EQ(out_text[2], input[2]);
  EXPECT_EQ(out_text[3], input[3]);
  EXPECT_EQ(out_text[4], input[4]);
  EXPECT_EQ(out_text[5], input[5]);
  EXPECT_EQ(out_text[6], input[6]);
  EXPECT_EQ(out_text[7], input[7]);
}
