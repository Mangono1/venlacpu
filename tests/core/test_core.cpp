#include <cassert>
#include <iostream>

#include "venla/core/device.hpp"
#include "venla/core/dtype.hpp"
#include "venla/core/shape.hpp"
#include "venla/core/stride.hpp"
#include "venla/core/version.hpp"

int main() {
    assert(venla::VERSION_MAJOR == 2);
    assert(venla::VERSION_MINOR == 2);
    assert(venla::VERSION_PATCH == 0);

    assert(
        venla::dtype_size(venla::DType::Float32) == 4
    );

    assert(
        venla::dtype_size(venla::DType::Float64) == 8
    );

    venla::Shape shape({2, 3, 4});

    assert(shape.ndim() == 3);
    assert(shape.rank() == 3);
    assert(shape.numel() == 24);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    assert(shape[2] == 4);

    venla::Stride stride(shape);

    assert(stride.ndim() == 3);
    assert(stride[0] == 12);
    assert(stride[1] == 4);
    assert(stride[2] == 1);
    assert(stride.is_contiguous(shape));

    venla::Device device = venla::Device::cpu();

    assert(device.is_cpu());
    assert(device.to_string() == "cpu");

    std::cout << "VENLACPU core tests passed\n";

    return 0;
}
