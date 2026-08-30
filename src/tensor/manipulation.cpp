#include "venla/tensor/manipulation.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void require_contiguous(
    const Tensor& input,
    const char* operation
) {
    if (!input.is_contiguous()) {
        throw std::runtime_error(
            std::string(operation) +
            ": non-contiguous tensors are not supported"
        );
    }
}

void require_same_dtype(
    const Tensor& a,
    const Tensor& b,
    const char* operation
) {
    if (a.dtype() != b.dtype()) {
        std::ostringstream message;

        message
            << operation
            << ": dtype mismatch: "
            << dtype_name(a.dtype())
            << " vs "
            << dtype_name(b.dtype());

        throw std::runtime_error(
            message.str()
        );
    }
}

void require_same_device(
    const Tensor& a,
    const Tensor& b,
    const char* operation
) {
    if (a.device().type() != b.device().type()) {
        throw std::runtime_error(
            std::string(operation) +
            ": tensors must be on the same device"
        );
    }
}

// ============================================================
// LINEAR INDEX HELPERS
// ============================================================

std::vector<std::size_t> unravel_index(
    std::size_t index,
    const Shape& shape
) {
    std::vector<std::size_t> coordinates(
        shape.ndim(),
        0
    );

    if (shape.ndim() == 0) {
        return coordinates;
    }

    for (std::size_t i = shape.ndim(); i > 0; --i) {
        const std::size_t dim = i - 1;

        coordinates[dim] =
            index % shape[dim];

        index /= shape[dim];
    }

    return coordinates;
}

std::size_t ravel_index(
    const std::vector<std::size_t>& coordinates,
    const Shape& shape
) {
    if (coordinates.size() != shape.ndim()) {
        throw std::runtime_error(
            "ravel_index: rank mismatch"
        );
    }

    std::size_t index = 0;

    for (std::size_t i = 0;
         i < shape.ndim();
         ++i) {

        if (coordinates[i] >= shape[i]) {
            throw std::out_of_range(
                "ravel_index: coordinate out of range"
            );
        }

        index =
            index * shape[i] +
            coordinates[i];
    }

    return index;
}

// ============================================================
// COPY
// ============================================================

Tensor copy_tensor(
    const Tensor& input
) {
    Tensor result =
        Tensor::empty(
            input.shape(),
            input.dtype(),
            input.device()
        );

    if (input.nbytes() != 0) {
        std::memcpy(
            result.data(),
            input.data(),
            input.nbytes()
        );
    }

    return result;
}

// ============================================================
// PRODUCT
// ============================================================

std::size_t product(
    const std::vector<std::size_t>& values,
    std::size_t begin,
    std::size_t end
) {
    std::size_t result = 1;

    for (std::size_t i = begin;
         i < end;
         ++i) {

        result *= values[i];
    }

    return result;
}

} // namespace

// ============================================================
// RESHAPE
// ============================================================

Tensor reshape(
    const Tensor& input,
    const Shape& new_shape
) {
    require_contiguous(
        input,
        "reshape"
    );

    if (input.numel() != new_shape.numel()) {
        std::ostringstream message;

        message
            << "reshape: cannot reshape "
            << input.shape().to_string()
            << " with "
            << input.numel()
            << " elements into "
            << new_shape.to_string()
            << " with "
            << new_shape.numel()
            << " elements";

        throw std::runtime_error(
            message.str()
        );
    }

    Tensor result =
        Tensor::empty(
            new_shape,
            input.dtype(),
            input.device()
        );

    if (input.nbytes() != 0) {
        std::memcpy(
            result.data(),
            input.data(),
            input.nbytes()
        );
    }

    return result;
}

Tensor reshape(
    const Tensor& input,
    std::initializer_list<std::size_t> shape
) {
    return reshape(
        input,
        Shape(shape)
    );
}

// ============================================================
// FLATTEN
// ============================================================

Tensor flatten(
    const Tensor& input
) {
    return reshape(
        input,
        Shape{input.numel()}
    );
}

Tensor flatten(
    const Tensor& input,
    std::size_t start_dim,
    std::size_t end_dim
) {
    const std::size_t rank =
        input.ndim();

    if (rank == 0) {
        throw std::runtime_error(
            "flatten: scalar tensor has no dimensions"
        );
    }

    if (start_dim >= rank ||
        end_dim >= rank) {

        throw std::out_of_range(
            "flatten: dimension out of range"
        );
    }

    if (start_dim > end_dim) {
        throw std::runtime_error(
            "flatten: start_dim must be <= end_dim"
        );
    }

    std::vector<std::size_t> dimensions;

    for (std::size_t i = 0;
         i < start_dim;
         ++i) {

        dimensions.push_back(
            input.shape()[i]
        );
    }

    std::size_t flattened_size = 1;

    for (std::size_t i = start_dim;
         i <= end_dim;
         ++i) {

        flattened_size *=
            input.shape()[i];
    }

    dimensions.push_back(
        flattened_size
    );

    for (std::size_t i = end_dim + 1;
         i < rank;
         ++i) {

        dimensions.push_back(
            input.shape()[i]
        );
    }

    return reshape(
        input,
        Shape(dimensions)
    );
}

// ============================================================
// SQUEEZE
// ============================================================

Tensor squeeze(
    const Tensor& input
) {
    std::vector<std::size_t> dimensions;

    for (std::size_t i = 0;
         i < input.ndim();
         ++i) {

        if (input.shape()[i] != 1) {
            dimensions.push_back(
                input.shape()[i]
            );
        }
    }

    return reshape(
        input,
        Shape(dimensions)
    );
}

Tensor squeeze(
    const Tensor& input,
    std::size_t dim
) {
    if (dim >= input.ndim()) {
        throw std::out_of_range(
            "squeeze: dimension out of range"
        );
    }

    if (input.shape()[dim] != 1) {
        throw std::runtime_error(
            "squeeze: selected dimension must have size 1"
        );
    }

    std::vector<std::size_t> dimensions;

    for (std::size_t i = 0;
         i < input.ndim();
         ++i) {

        if (i != dim) {
            dimensions.push_back(
                input.shape()[i]
            );
        }
    }

    return reshape(
        input,
        Shape(dimensions)
    );
}

// ============================================================
// UNSQUEEZE
// ============================================================

Tensor unsqueeze(
    const Tensor& input,
    std::size_t dim
) {
    if (dim > input.ndim()) {
        throw std::out_of_range(
            "unsqueeze: dimension out of range"
        );
    }

    std::vector<std::size_t> dimensions;

    dimensions.reserve(
        input.ndim() + 1
    );

    for (std::size_t i = 0;
         i < input.ndim() + 1;
         ++i) {

        if (i == dim) {
            dimensions.push_back(1);
        }
        else {
            const std::size_t source_dim =
                i < dim
                    ? i
                    : i - 1;

            dimensions.push_back(
                input.shape()[source_dim]
            );
        }
    }

    return reshape(
        input,
        Shape(dimensions)
    );
}

// ============================================================
// CONCATENATE
// ============================================================

Tensor concatenate(
    const std::vector<Tensor>& tensors,
    std::size_t dim
) {
    if (tensors.empty()) {
        throw std::runtime_error(
            "concatenate: tensor list cannot be empty"
        );
    }

    const Tensor& first =
        tensors[0];

    require_contiguous(
        first,
        "concatenate"
    );

    const std::size_t rank =
        first.ndim();

    if (rank == 0) {
        throw std::runtime_error(
            "concatenate: scalar tensors are not supported"
        );
    }

    if (dim >= rank) {
        throw std::out_of_range(
            "concatenate: dimension out of range"
        );
    }

    std::size_t output_dimension =
        0;

    for (const Tensor& tensor : tensors) {

        require_contiguous(
            tensor,
            "concatenate"
        );

        require_same_dtype(
            first,
            tensor,
            "concatenate"
        );

        require_same_device(
            first,
            tensor,
            "concatenate"
        );

        if (tensor.ndim() != rank) {
            throw std::runtime_error(
                "concatenate: all tensors must have the same rank"
            );
        }

        for (std::size_t d = 0;
             d < rank;
             ++d) {

            if (d == dim) {
                continue;
            }

            if (tensor.shape()[d] !=
                first.shape()[d]) {

                throw std::runtime_error(
                    "concatenate: non-concatenated dimensions must match"
                );
            }
        }

        output_dimension +=
            tensor.shape()[dim];
    }

    std::vector<std::size_t>
        output_dimensions =
            first.shape().dimensions();

    output_dimensions[dim] =
        output_dimension;

    Tensor result =
        Tensor::empty(
            Shape(output_dimensions),
            first.dtype(),
            first.device()
        );

    const std::size_t element_size =
        dtype_size(first.dtype());

    std::size_t destination_offset =
        0;

    // --------------------------------------------------------
    // Copy using outer / concat / inner regions.
    // --------------------------------------------------------

    const std::size_t outer =
        product(
            first.shape().dimensions(),
            0,
            dim
        );

    const std::size_t inner =
        product(
            first.shape().dimensions(),
            dim + 1,
            rank
        );

    std::size_t destination =
        0;

    for (std::size_t outer_index = 0;
         outer_index < outer;
         ++outer_index) {

        for (const Tensor& tensor : tensors) {

            const std::size_t chunk =
                tensor.shape()[dim] *
                inner;

            const std::size_t source_index =
                outer_index *
                chunk;

            std::memcpy(
                static_cast<char*>(result.data()) +
                    destination *
                    element_size,
                static_cast<const char*>(tensor.data()) +
                    source_index *
                    element_size,
                chunk *
                    element_size
            );

            destination += chunk;
        }
    }

    (void)destination_offset;

    return result;
}

// ============================================================
// STACK
// ============================================================

Tensor stack(
    const std::vector<Tensor>& tensors,
    std::size_t dim
) {
    if (tensors.empty()) {
        throw std::runtime_error(
            "stack: tensor list cannot be empty"
        );
    }

    const Tensor& first =
        tensors[0];

    if (dim > first.ndim()) {
        throw std::out_of_range(
            "stack: dimension out of range"
        );
    }

    for (const Tensor& tensor : tensors) {

        require_contiguous(
            tensor,
            "stack"
        );

        require_same_dtype(
            first,
            tensor,
            "stack"
        );

        require_same_device(
            first,
            tensor,
            "stack"
        );

        if (tensor.shape() != first.shape()) {
            throw std::runtime_error(
                "stack: all tensors must have identical shapes"
            );
        }
    }

    std::vector<std::size_t>
        output_dimensions;

    output_dimensions.reserve(
        first.ndim() + 1
    );

    for (std::size_t i = 0;
         i < first.ndim() + 1;
         ++i) {

        if (i == dim) {
            output_dimensions.push_back(
                tensors.size()
            );
        }
        else {
            const std::size_t source_dim =
                i < dim
                    ? i
                    : i - 1;

            output_dimensions.push_back(
                first.shape()[source_dim]
            );
        }
    }

    Tensor result =
        Tensor::empty(
            Shape(output_dimensions),
            first.dtype(),
            first.device()
        );

    const std::size_t element_size =
        dtype_size(first.dtype());

    const std::size_t tensor_bytes =
        first.nbytes();

    const std::size_t outer =
        product(
            output_dimensions,
            0,
            dim
        );

    const std::size_t inner =
        product(
            output_dimensions,
            dim + 1,
            output_dimensions.size()
        );

    // Each stacked tensor occupies:
    //
    // outer × inner elements.
    //
    // Since the source tensor has the same
    // shape, this is exactly its number of elements.
    const std::size_t source_elements =
        first.numel();

    if (outer * inner != source_elements) {
        throw std::runtime_error(
            "stack: internal shape calculation error"
        );
    }

    char* destination =
        static_cast<char*>(result.data());

    for (std::size_t outer_index = 0;
         outer_index < outer;
         ++outer_index) {

        for (std::size_t tensor_index = 0;
             tensor_index < tensors.size();
             ++tensor_index) {

            const std::size_t source_offset =
                outer_index *
                inner;

            const std::size_t destination_offset =
                (
                    outer_index *
                    tensors.size() *
                    inner
                ) +
                tensor_index *
                inner;

            std::memcpy(
                destination +
                    destination_offset *
                    element_size,
                static_cast<const char*>(
                    tensors[tensor_index].data()
                ) +
                    source_offset *
                    element_size,
                inner *
                    element_size
            );
        }
    }

    (void)tensor_bytes;

    return result;
}

// ============================================================
// INDEX
// ============================================================

Tensor index(
    const Tensor& input,
    const std::vector<std::size_t>& indices
) {
    require_contiguous(
        input,
        "index"
    );

    if (indices.size() != input.ndim()) {
        std::ostringstream message;

        message
            << "index: expected "
            << input.ndim()
            << " indices, got "
            << indices.size();

        throw std::runtime_error(
            message.str()
        );
    }

    const std::size_t linear_index =
        ravel_index(
            indices,
            input.shape()
        );

    Tensor result =
        Tensor::empty(
            Shape{},
            input.dtype(),
            input.device()
        );

    const std::size_t element_size =
        dtype_size(input.dtype());

    std::memcpy(
        result.data(),
        static_cast<const char*>(input.data()) +
            linear_index *
            element_size,
        element_size
    );

    return result;
}

Tensor index(
    const Tensor& input,
    std::size_t position
) {
    if (input.ndim() != 1) {
        throw std::runtime_error(
            "index(position): tensor must be 1D"
        );
    }

    return index(
        input,
        std::vector<std::size_t>{position}
    );
}

// ============================================================
// SLICE
// ============================================================

Tensor slice(
    const Tensor& input,
    std::size_t dim,
    std::size_t start,
    std::size_t stop,
    std::size_t step
) {
    require_contiguous(
        input,
        "slice"
    );

    if (input.ndim() == 0) {
        throw std::runtime_error(
            "slice: scalar tensors cannot be sliced"
        );
    }

    if (dim >= input.ndim()) {
        throw std::out_of_range(
            "slice: dimension out of range"
        );
    }

    if (step == 0) {
        throw std::runtime_error(
            "slice: step cannot be zero"
        );
    }

    const std::size_t dimension_size =
        input.shape()[dim];

    if (start > dimension_size ||
        stop > dimension_size) {

        throw std::out_of_range(
            "slice: start/stop out of range"
        );
    }

    if (start > stop) {
        throw std::runtime_error(
            "slice: start must be <= stop"
        );
    }

    std::size_t slice_size = 0;

    if (start < stop) {
        slice_size =
            (
                stop -
                start +
                step -
                1
            ) /
            step;
    }

    std::vector<std::size_t>
        output_dimensions =
            input.shape().dimensions();

    output_dimensions[dim] =
        slice_size;

    Tensor result =
        Tensor::empty(
            Shape(output_dimensions),
            input.dtype(),
            input.device()
        );

    const std::size_t element_size =
        dtype_size(input.dtype());

    for (std::size_t output_index = 0;
         output_index < result.numel();
         ++output_index) {

        std::vector<std::size_t>
            output_coordinates =
                unravel_index(
                    output_index,
                    result.shape()
                );

        std::vector<std::size_t>
            input_coordinates =
                output_coordinates;

        input_coordinates[dim] =
            start +
            output_coordinates[dim] *
            step;

        const std::size_t input_index =
            ravel_index(
                input_coordinates,
                input.shape()
            );

        std::memcpy(
            static_cast<char*>(result.data()) +
                output_index *
                element_size,
            static_cast<const char*>(input.data()) +
                input_index *
                element_size,
            element_size
        );
    }

    return result;
}

} // namespace venla
