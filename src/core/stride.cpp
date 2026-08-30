#include "venla/core/stride.hpp"

#include <sstream>
#include <stdexcept>

namespace venla {

Stride::Stride() = default;

Stride::Stride(const Shape& shape)
    : values_(contiguous(shape).values()) {
}

Stride::Stride(const std::vector<std::size_t>& values)
    : values_(values) {
}

Stride Stride::contiguous(const Shape& shape) {
    std::vector<std::size_t> result(shape.ndim(), 1);

    if (shape.ndim() == 0) {
        return Stride(result);
    }

    std::size_t running = 1;

    for (std::size_t i = shape.ndim(); i > 0; --i) {
        const std::size_t dimension = i - 1;

        result[dimension] = running;

        running *= shape[dimension];
    }

    return Stride(result);
}

const std::vector<std::size_t>& Stride::values() const {
    return values_;
}

std::size_t Stride::ndim() const {
    return values_.size();
}

std::size_t Stride::operator[](std::size_t index) const {
    if (index >= values_.size()) {
        throw std::out_of_range("Stride index out of range");
    }

    return values_[index];
}

bool Stride::is_contiguous(const Shape& shape) const {
    if (shape.ndim() != values_.size()) {
        return false;
    }

    const Stride expected = Stride::contiguous(shape);

    if (values_.size() != expected.values_.size()) {
        return false;
    }

    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (values_[i] != expected.values_[i]) {
            return false;
        }
    }

    return true;
}

std::string Stride::to_string() const {
    std::ostringstream out;

    out << "[";

    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }

        out << values_[i];
    }

    out << "]";

    return out.str();
}

} // namespace venla
