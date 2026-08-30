#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace venla {

enum class DType {
    Bool,

    Int8,
    Int16,
    Int32,
    Int64,

    UInt8,
    UInt16,
    UInt32,
    UInt64,

    Float16,
    BFloat16,
    Float32,
    Float64,

    Complex64,
    Complex128
};

const char* dtype_name(DType dtype);

std::size_t dtype_size(DType dtype);

bool dtype_is_floating(DType dtype);

bool dtype_is_integral(DType dtype);

bool dtype_is_complex(DType dtype);

} // namespace venla
