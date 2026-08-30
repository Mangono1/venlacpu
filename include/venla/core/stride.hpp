#pragma once

#include "venla/core/shape.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace venla {

class Stride {
public:
    Stride();

    explicit Stride(const Shape& shape);

    explicit Stride(const std::vector<std::size_t>& values);

    const std::vector<std::size_t>& values() const;

    std::size_t ndim() const;

    std::size_t operator[](std::size_t index) const;

    bool is_contiguous(const Shape& shape) const;

    std::string to_string() const;

    static Stride contiguous(const Shape& shape);

private:
    std::vector<std::size_t> values_;
};

} // namespace venla
