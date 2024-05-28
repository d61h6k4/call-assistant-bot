#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/formats/yuv_image.h"
#include "screenreader/utils/ffmpeg.h"

namespace aikit {

// This Calculator captures screen and produces video packets.
//
// Output Streams:
//   VIDEO: Output video frames (YUVImage).
//
// Example config:
// node {
//   calculator: "FFMPEGCaptureScreenCalculator"
//   output_stream: "VIDEO:video_frames"
//   output_stream: "VIDEO_PRESTREAM:video_header"
// }
class FFMPEGCaptureScreenCalculator : public mediapipe::api2::Node {
public:
  static constexpr mediapipe::api2::Output<mediapipe::YUVImage> kOutVideo{
      "VIDEO"};
  static constexpr mediapipe::api2::Output<mediapipe::VideoHeader>
      kOutVideoPrestream{"VIDEO_PRESTREAM"};

  MEDIAPIPE_NODE_CONTRACT(
      kOutVideo, kOutVideoPrestream,
      mediapipe::api2::StreamHandler("ImmediateInputStreamHandler"),
      mediapipe::api2::TimestampChange::Arbitrary());

  absl::Status Open(mediapipe::CalculatorContext *cc) override;
  absl::Status Process(mediapipe::CalculatorContext *cc) override;
  absl::Status Close(mediapipe::CalculatorContext *cc) override;

private:
  utils::VideoStreamContext video_stream_context_;
  mediapipe::Timestamp prev_image_timestamp_ = mediapipe::Timestamp::Unset();
  mediapipe::Timestamp prev_audio_timestamp_ = mediapipe::Timestamp::Unset();

  // https://ffmpeg.org/doxygen/trunk/structAVFrame.html
  AVFrame *frame_ = nullptr;

  // https://ffmpeg.org/doxygen/trunk/structAVPacket.html
  AVPacket *packet_ = nullptr;
};
MEDIAPIPE_REGISTER_NODE(FFMPEGCaptureScreenCalculator);

absl::Status
FFMPEGCaptureScreenCalculator::Open(mediapipe::CalculatorContext *cc) {

#if __APPLE__
  auto video_stream_context_or = utils::CaptureDevice("avfoundation", "3:");
#elif __linux__
  auto video_stream_context_or =
      utils::CaptureDevice("x11grab", ":0.0+100,200");
#endif

  if (!video_stream_context_or.ok()) {
    if (absl::IsFailedPrecondition(video_stream_context_or.status())) {
      ABSL_LOG(WARNING) << "Video stream context was not fully initialized. "
                        << video_stream_context_or.status().message();
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << video_stream_context_or.status().message();
    }
  }

  video_stream_context_ = video_stream_context_or.value();

  if (!video_stream_context_.image_stream_context.has_value()) {
    return mediapipe::InvalidArgumentError(
        "Video stream does not contain image stream. Stop processing.");
  }

  auto header = absl::make_unique<mediapipe::VideoHeader>();
  header->format = mediapipe::ImageFormat::YCBCR420P;
  header->width = video_stream_context_.image_stream_context->width;
  header->height = video_stream_context_.image_stream_context->height;
  header->frame_rate = video_stream_context_.image_stream_context->frame_rate;
  header->duration = 1;

  kOutVideoPrestream(cc).Send(
      mediapipe::api2::PacketAdopting<mediapipe::VideoHeader>(std::move(header))
          .At(mediapipe::Timestamp::PreStream()));
  kOutVideoPrestream(cc).Close();

  frame_ = av_frame_alloc();
  if (!frame_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVFrame";
  }

  packet_ = av_packet_alloc();
  if (!packet_) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "failed to allocate memory for AVPacket";
  }

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureScreenCalculator::Close(mediapipe::CalculatorContext *cc) {
  av_packet_free(&packet_);
  av_frame_free(&frame_);

  utils::DestroyVideoStreamContext(video_stream_context_);

  return absl::OkStatus();
}

absl::Status
FFMPEGCaptureScreenCalculator::Process(mediapipe::CalculatorContext *cc) {
  if (!video_stream_context_.image_stream_context.has_value()) {
    ABSL_LOG(ERROR) << "There is no image context. Stop processing.";
    return mediapipe::tool::StatusStop();
  }

  // fill the Packet with data from the Stream
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
  while (av_read_frame(video_stream_context_.format_context, packet_) >= 0) {
    // if it's the video stream
    if (packet_->stream_index ==
        video_stream_context_.image_stream_context->stream_index) {

      auto s = utils::PacketToFrame(
          video_stream_context_.image_stream_context->codec_context, packet_,
          frame_);

      if (s.ok()) {
        ABSL_LOG_FIRST_N(INFO, 1)
            << "Start processing. First frame "
            << video_stream_context_.image_stream_context->codec_context
                   ->frame_num
            << "(type=" << av_get_picture_type_char(frame_->pict_type)
            << ", size=" << frame_->pkt_size
            << " bytes, format=" << frame_->format << ") pts " << frame_->pts
            << " key_frame " << frame_->key_frame << "[DTS " << frame_->pkt_dts
            << "]"
            << "Context: " << video_stream_context_.image_stream_context->width
            << "x" << video_stream_context_.image_stream_context->height;

        // Use microsecond as the unit of time.
        mediapipe::Timestamp timestamp(static_cast<int64_t>(
            static_cast<float>(
                video_stream_context_.image_stream_context->start_time) +
            video_stream_context_.image_stream_context->time_base *
                static_cast<float>(frame_->pts) * 1000000.0f));

        // If the timestamp of the current frame is not greater than the one
        // of the previous frame, the new frame will be discarded.
        if (prev_image_timestamp_ < timestamp) {
          auto yuv_or = utils::ReadImageFrame(
              video_stream_context_.image_stream_context.value(), frame_);
          kOutVideo(cc).Send(
              mediapipe::api2::PacketAdopting<mediapipe::YUVImage>(
                  new mediapipe::YUVImage(
                      libyuv::FOURCC_I420, std::move(yuv_or->y),
                      video_stream_context_.image_stream_context->width,
                      std::move(yuv_or->u),
                      (video_stream_context_.image_stream_context->width + 1) /
                          2,
                      std::move(yuv_or->v),
                      (video_stream_context_.image_stream_context->width + 1) /
                          2,
                      video_stream_context_.image_stream_context->width,
                      video_stream_context_.image_stream_context->height))
                  .At(timestamp));

          prev_image_timestamp_ = timestamp;
          // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
          av_packet_unref(packet_);
          return absl::OkStatus();
        } else {
          ABSL_LOG(WARNING) << "Unmonotonic timestamps "
                            << prev_image_timestamp_ << " and " << timestamp;
        }
      }
    }
    // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
    av_packet_unref(packet_);
  }

  ABSL_LOG(INFO) << "Got last frame";
  return mediapipe::tool::StatusStop();
}

} // namespace aikit
