#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace venla {

class Shape {
public:
    Shape();

    Shape(
        std::initializer_list<std::size_t> dimensions
    );

    explicit Shape(
        const std::vector<std::size_t>& dimensions
    );

    std::size_t ndim() const;

    std::size_t rank() const;

    std::size_t numel() const;

    const std::vector<std::size_t>& dimensions() const;

    std::size_t operator[](
        std::size_t index
    ) const;

    bool operator==(
        const Shape& other
    ) const;

    bool operator!=(
        const Shape& other
    ) const;

    std::string to_string() const;

private:
    std::vector<std::size_t> dimensions_;
};

} // namespace venla
