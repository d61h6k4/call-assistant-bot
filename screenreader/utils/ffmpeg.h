#pragma once

#include "absl/status/status.h"

namespace aikit {
namespace utils {
// Captures data from the device.
// This operation is operating system dependent:
//  MacOS: device_name is avfoundation
//         driver_url is ":<audio_device_index>" or "<screen_device_index>:"
//    to find audio_device_index please execute the next command
//    ffmpeg -f avfoundation -list_devices true -i ""
//
//  Linux: device_name is x11grab
//         driver_url is alsa/pulse for audio and x11grab for screen
absl::Status CaptureDevice(const std::string &device_name,
                          const std::string &driver_url);
} // namespace utils
} // namespace aikit
