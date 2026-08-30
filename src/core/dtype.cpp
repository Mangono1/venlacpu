#include "venla/core/dtype.hpp"

#include <stdexcept>

namespace venla {

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool:       return "bool";

        case DType::Int8:       return "int8";
        case DType::Int16:      return "int16";
        case DType::Int32:      return "int32";
        case DType::Int64:      return "int64";

        case DType::UInt8:      return "uint8";
        case DType::UInt16:     return "uint16";
        case DType::UInt32:     return "uint32";
        case DType::UInt64:     return "uint64";

        case DType::Float16:    return "float16";
        case DType::BFloat16:   return "bfloat16";
        case DType::Float32:    return "float32";
        case DType::Float64:    return "float64";

        case DType::Complex64:  return "complex64";
        case DType::Complex128: return "complex128";
    }

    return "unknown";
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Bool:
        case DType::Int8:
        case DType::UInt8:
            return 1;

        case DType::Int16:
        case DType::UInt16:
        case DType::Float16:
        case DType::BFloat16:
            return 2;

        case DType::Int32:
        case DType::UInt32:
        case DType::Float32:
            return 4;

        case DType::Int64:
        case DType::UInt64:
        case DType::Float64:
        case DType::Complex64:
            return 8;

        case DType::Complex128:
            return 16;
    }

    throw std::runtime_error("Unknown DType");
}

bool dtype_is_floating(DType dtype) {
    return dtype == DType::Float16 ||
           dtype == DType::BFloat16 ||
           dtype == DType::Float32 ||
           dtype == DType::Float64;
}

bool dtype_is_integral(DType dtype) {
    return dtype == DType::Int8 ||
           dtype == DType::Int16 ||
           dtype == DType::Int32 ||
           dtype == DType::Int64 ||
           dtype == DType::UInt8 ||
           dtype == DType::UInt16 ||
           dtype == DType::UInt32 ||
           dtype == DType::UInt64;
}

bool dtype_is_complex(DType dtype) {
    return dtype == DType::Complex64 ||
           dtype == DType::Complex128;
}

} // namespace venla
