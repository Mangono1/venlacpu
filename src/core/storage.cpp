#include "venla/core/storage.hpp"

namespace venla {

Storage::Storage()
    : data_(nullptr),
      bytes_(0) {
}

Storage::Storage(std::size_t bytes)
    : data_(bytes == 0 ? nullptr : std::make_unique<unsigned char[]>(bytes)),
      bytes_(bytes) {
}

Storage::~Storage() = default;

void* Storage::data() {
    return data_.get();
}

const void* Storage::data() const {
    return data_.get();
}

std::size_t Storage::bytes() const {
    return bytes_;
}

bool Storage::empty() const {
    return bytes_ == 0;
}

} // namespace venla
