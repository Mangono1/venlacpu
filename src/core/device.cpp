#include "venla/core/device.hpp"

namespace venla {

Device::Device()
    : type_(DeviceType::CPU) {
}

Device::Device(DeviceType type)
    : type_(type) {
}

Device Device::cpu() {
    return Device(DeviceType::CPU);
}

DeviceType Device::type() const {
    return type_;
}

bool Device::is_cpu() const {
    return type_ == DeviceType::CPU;
}

std::string Device::to_string() const {
    return "cpu";
}

} // namespace venla
