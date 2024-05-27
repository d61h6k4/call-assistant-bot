
#include "screenreader/utils/ffmpeg.h"
#include "gtest/gtest.h"

TEST(TestFFmpegUtils, CheckCaptureAudio) {
  auto res = aikit::utils::CaptureDevice("avfoundation", ":0");
  EXPECT_TRUE(res.ok()) << res.status().message();
  EXPECT_TRUE(res->audio_stream_context.has_value());
  aikit::utils::DestroyVideoStreamContext(res.value());
}

TEST(TestFFmpegUtils, CheckCaptureScreen) {
  auto res = aikit::utils::CaptureDevice("avfoundation", "2:");
  EXPECT_TRUE(res.ok()) << res.status().message();

  aikit::utils::DestroyVideoStreamContext(res.value());
}

TEST(TestFFmpegUtils, PacketToFramCheck) {
  auto res = aikit::utils::CaptureDevice("avfoundation", ":2");
  EXPECT_TRUE(res.ok()) << res.status().message();

  AVFrame *frame = av_frame_alloc();
  EXPECT_TRUE(frame != nullptr);

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet != nullptr);

  int n = 3;
  while (n > 0 && av_read_frame(res->format_context, packet) == 0) {
    --n;
    if (packet->stream_index == res->image_stream_context->stream_index) {
      auto s = aikit::utils::PacketToFrame(
          res->image_stream_context->codec_context, packet, frame);

      if (!s.ok()) {
        EXPECT_TRUE(absl::IsFailedPrecondition(s));
      }
    } else if (packet->stream_index ==
               res->audio_stream_context->stream_index) {
      auto s = aikit::utils::PacketToFrame(
          res->audio_stream_context->codec_context, packet, frame);

      EXPECT_TRUE(s.ok());
    }
  }

  av_packet_free(&packet);
  av_frame_free(&frame);
  aikit::utils::DestroyVideoStreamContext(res.value());
}

TEST(TestFFmpegUtils, ReadAudioFrameCheck) {
  auto res = aikit::utils::CaptureDevice("avfoundation", ":2");
  EXPECT_TRUE(res.ok());

  AVFrame *frame = av_frame_alloc();
  EXPECT_TRUE(frame != nullptr);

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet != nullptr);

  int n = 3;
  std::vector<float> audio_track;
  while (n > 0 && av_read_frame(res->format_context, packet) == 0) {
    --n;
    if (packet->stream_index == res->audio_stream_context->stream_index) {
      auto s = aikit::utils::PacketToFrame(
          res->audio_stream_context->codec_context, packet, frame);

      EXPECT_TRUE(s.ok());

      auto audio_data =
          ReadAudioFrame(res->audio_stream_context.value(), frame);
      EXPECT_TRUE(audio_data.ok());

      EXPECT_LE(audio_data->size(), frame->nb_samples);
      audio_track.insert(audio_track.end(), audio_data->begin(),
                         audio_data->end());
    }
  }

  av_packet_free(&packet);
  av_frame_free(&frame);
  aikit::utils::DestroyVideoStreamContext(res.value());
}

TEST(TestFFmpegUtils, ReadImageFrameCheck) {
  auto res = aikit::utils::CaptureDevice("avfoundation", "3:");
  EXPECT_TRUE(res.ok());

  AVFrame *frame = av_frame_alloc();
  EXPECT_TRUE(frame != nullptr);

  AVPacket *packet = av_packet_alloc();
  EXPECT_TRUE(packet != nullptr);

  int n = 3;
  while (n > 0 && av_read_frame(res->format_context, packet) == 0) {
    --n;
    if (packet->stream_index == res->image_stream_context->stream_index) {
      auto s = aikit::utils::PacketToFrame(
          res->image_stream_context->codec_context, packet, frame);

      if (!s.ok()) {
        EXPECT_TRUE(absl::IsFailedPrecondition(s));
      } else {

        auto image_data = aikit::utils::ReadImageFrame(
            res->image_stream_context.value(), frame);
        EXPECT_TRUE(image_data.ok()) << image_data.status().message();
      }
    }
  }

  av_packet_free(&packet);
  av_frame_free(&frame);
  aikit::utils::DestroyVideoStreamContext(res.value());
}
