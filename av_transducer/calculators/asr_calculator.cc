#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "ml/asr/model.h"
#include "av_transducer/utils/audio.h"
#include <vector>

namespace aikit {

class ASRCalculator : public mediapipe::api2::Node {
public:
    static constexpr mediapipe::api2::SideInput<std::string> kInASRModelPath{
        "ASR_MODEL_PATH"};
    static constexpr mediapipe::api2::SideInput<std::string> kInSPKModelPath{
        "SPK_MODEL_PATH"};
    static constexpr mediapipe::api2::SideInput<int>::Optional kInBufferDurationSec{
        "BUFFER_DURATION_SEC"};
    static constexpr mediapipe::api2::Input<media::AudioFrame> kInAudio{
        "AUDIO"};
    static constexpr mediapipe::api2::Output<aikit::ml::ASRResult>
        kOutASRResult{"ASR_RESULT"};
    MEDIAPIPE_NODE_CONTRACT(kInASRModelPath, kInSPKModelPath, kInBufferDurationSec, kInAudio, kOutASRResult);

    absl::Status Open(mediapipe::CalculatorContext *cc) override;
    absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
    std::unique_ptr<aikit::ml::ASRModel> model_;
    std::vector<float> audio_buffer_;
    size_t buffer_size_;
    static constexpr size_t kSampleRate = 16000;
    static constexpr size_t kDefaultBufferDurationSec = 10;
};

MEDIAPIPE_REGISTER_NODE(ASRCalculator);

absl::Status ASRCalculator::Open(mediapipe::CalculatorContext *cc) {
    const std::string &asr_model_path = kInASRModelPath(cc).Get();
    const std::string &spk_model_path = kInSPKModelPath(cc).Get();
    model_ = std::make_unique<ml::ASRModel>(asr_model_path, spk_model_path);
    int buffer_duration_sec = kDefaultBufferDurationSec;
    if (kInBufferDurationSec(cc).IsConnected() && !kInBufferDurationSec(cc).IsEmpty()) {
        buffer_duration_sec = kInBufferDurationSec(cc).Get();
    }
    buffer_size_ = kSampleRate * buffer_duration_sec;
    audio_buffer_.reserve(buffer_size_);
    return absl::OkStatus();
}

absl::Status ASRCalculator::Process(mediapipe::CalculatorContext *cc) {
    const auto &audio_frame = kInAudio(cc).Get();
    auto status = audio_frame.AppendAudioData(audio_buffer_);
    if (!status.ok()) {
        return status;
    }
    if (audio_buffer_.size() >= buffer_size_) {
        auto result = model_->operator()(audio_buffer_);
        audio_buffer_.clear();
        if (!result.ok()) {
            return absl::OkStatus();
        }
        else {
            kOutASRResult(cc).Send(result.value());
        }
    }
    return absl::OkStatus();
}

} // namespace aikit