#include "venla/core/shape.hpp"

#include <sstream>
#include <stdexcept>

namespace venla {

// ============================================================
// CONSTRUCTORS
// ============================================================

Shape::Shape()
    : dimensions_() {
}

Shape::Shape(
    std::initializer_list<std::size_t> dimensions
)
    : dimensions_(dimensions) {
}

Shape::Shape(
    const std::vector<std::size_t>& dimensions
)
    : dimensions_(dimensions) {
}

// ============================================================
// DIMENSIONS
// ============================================================

std::size_t Shape::ndim() const {
    return dimensions_.size();
}

std::size_t Shape::rank() const {
    return dimensions_.size();
}

// ============================================================
// NUMBER OF ELEMENTS
//
// Shape{} represents a scalar.
//
// Shape{}       -> 1 element
// Shape{3}      -> 3 elements
// Shape{2, 3}   -> 6 elements
//
// ============================================================

std::size_t Shape::numel() const {

    if (dimensions_.empty()) {
        return 1;
    }

    std::size_t result = 1;

    for (const std::size_t dimension : dimensions_) {
        result *= dimension;
    }

    return result;
}

// ============================================================
// DIMENSIONS ACCESS
// ============================================================

const std::vector<std::size_t>&
Shape::dimensions() const {
    return dimensions_;
}

// ============================================================
// INDEX ACCESS
// ============================================================

std::size_t Shape::operator[](
    std::size_t index
) const {

    if (index >= dimensions_.size()) {
        throw std::out_of_range(
            "Shape index out of range"
        );
    }

    return dimensions_[index];
}

// ============================================================
// COMPARISON
// ============================================================

bool Shape::operator==(
    const Shape& other
) const {
    return dimensions_ == other.dimensions_;
}

bool Shape::operator!=(
    const Shape& other
) const {
    return !(*this == other);
}

// ============================================================
// STRING
// ============================================================

std::string Shape::to_string() const {

    std::ostringstream stream;

    stream << "[";

    for (std::size_t i = 0;
         i < dimensions_.size();
         ++i) {

        if (i > 0) {
            stream << ", ";
        }

        stream << dimensions_[i];
    }

    stream << "]";

    return stream.str();
}

} // namespace venla
