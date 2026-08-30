#pragma once

#include <cstddef>
#include <memory>

namespace venla {

class Storage {
public:
    Storage();

    explicit Storage(std::size_t bytes);

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;

    ~Storage();

    void* data();

    const void* data() const;

    std::size_t bytes() const;

    bool empty() const;

private:
    std::unique_ptr<unsigned char[]> data_;
    std::size_t bytes_;
};

} // namespace venla
