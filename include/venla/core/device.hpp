#pragma once

#include <string>

namespace venla {

enum class DeviceType {
    CPU
};

class Device {
public:
    Device();
    explicit Device(DeviceType type);

    static Device cpu();

    DeviceType type() const;

    bool is_cpu() const;

    std::string to_string() const;

private:
    DeviceType type_;
};

} // namespace venla
