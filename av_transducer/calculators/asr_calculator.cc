#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "ml/asr/model.h"
#include <vector>

namespace aikit {

class ASRCalculator : public mediapipe::api2::Node {
public:
    static constexpr mediapipe::api2::SideInput<std::string> kInASRModelPath{
        "ASR_MODEL_PATH"};
    static constexpr mediapipe::api2::SideInput<std::string> kInSPKModelPath{
        "SPK_MODEL_PATH"};
    static constexpr mediapipe::api2::SideInput<int> kInBufferDurationSec{
        "BUFFER_DURATION_SEC"};
    static constexpr mediapipe::api2::Input<std::vector<float>> kInAudio{"AUDIO"};
    static constexpr mediapipe::api2::Output<aikit::ml::ASRResult>
        kOutASRResult{"ASR_RESULT"};
    MEDIAPIPE_NODE_CONTRACT(kInASRModelPath, kInSPKModelPath, kInBufferDurationSec, kInAudio, kOutASRResult);

    absl::Status Open(mediapipe::CalculatorContext *cc) override;
    absl::Status Process(mediapipe::CalculatorContext *cc) override;

private:
    std::unique_ptr<aikit::ml::ASRModel> model_;
    std::vector<float> audio_buffer_;
    size_t buffer_position_ = 0;
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
    if (cc->InputSidePackets().HasTag(kInBufferDurationSec.Tag())) {
        buffer_duration_sec = kInBufferDurationSec(cc).Get();
    }
    buffer_size_ = kSampleRate * buffer_duration_sec;
    audio_buffer_.resize(buffer_size_);
    return absl::OkStatus();
}

absl::Status ASRCalculator::Process(mediapipe::CalculatorContext *cc) {
    const auto &audio = kInAudio(cc).Get();
  
    std::copy(audio.begin(), audio.end(), audio_buffer_.begin() + buffer_position_);
    buffer_position_ += audio.size();

    if (buffer_position_ >= buffer_size_) {
        auto result = model_->operator()(audio_buffer_);
        size_t remaining_samples = buffer_position_ - buffer_size_;
        std::copy(audio_buffer_.begin() + buffer_size_, audio_buffer_.end(), audio_buffer_.begin());
        buffer_position_ = remaining_samples;
        
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