#include "venla/math/operations.hpp"
#include "venla/autograd/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void require_float32(const Tensor& tensor, const char* operation) {
    if (tensor.dtype() != DType::Float32) {
        std::ostringstream message;

        message
            << operation
            << ": currently only Float32 is supported, got "
            << dtype_name(tensor.dtype());

        throw std::runtime_error(message.str());
    }

    if (!tensor.device().is_cpu()) {
        throw std::runtime_error(
            std::string(operation) +
            ": only CPU device is currently supported"
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

        throw std::runtime_error(message.str());
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
// BROADCASTING
// ============================================================

Shape broadcast_shape(
    const Shape& a,
    const Shape& b
) {
    const std::size_t rank_a = a.ndim();
    const std::size_t rank_b = b.ndim();

    const std::size_t output_rank =
        std::max(rank_a, rank_b);

    std::vector<std::size_t> dimensions(output_rank);

    for (std::size_t i = 0; i < output_rank; ++i) {
        const std::size_t from_a =
            i < output_rank - rank_a
                ? 1
                : a[i - (output_rank - rank_a)];

        const std::size_t from_b =
            i < output_rank - rank_b
                ? 1
                : b[i - (output_rank - rank_b)];

        if (from_a != from_b &&
            from_a != 1 &&
            from_b != 1) {

            std::ostringstream message;

            message
                << "Broadcast shape mismatch: "
                << a.to_string()
                << " vs "
                << b.to_string();

            throw std::runtime_error(message.str());
        }

        dimensions[i] =
            std::max(from_a, from_b);
    }

    return Shape(dimensions);
}

// Convert linear index into multidimensional coordinates.
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
        const std::size_t dimension = i - 1;

        coordinates[dimension] =
            index % shape[dimension];

        index /= shape[dimension];
    }

    return coordinates;
}

// Convert coordinates to contiguous linear index.
std::size_t ravel_index(
    const std::vector<std::size_t>& coordinates,
    const Shape& shape
) {
    if (coordinates.size() != shape.ndim()) {
        throw std::runtime_error(
            "Coordinate rank does not match shape rank"
        );
    }

    std::size_t index = 0;

    for (std::size_t i = 0; i < shape.ndim(); ++i) {
        index =
            index * shape[i] +
            coordinates[i];
    }

    return index;
}

// Map an output coordinate to a broadcasted input.
std::size_t broadcast_input_index(
    const std::vector<std::size_t>& output_coordinates,
    const Shape& input_shape
) {
    const std::size_t output_rank =
        output_coordinates.size();

    const std::size_t input_rank =
        input_shape.ndim();

    if (input_rank == 0) {
        return 0;
    }

    std::vector<std::size_t> input_coordinates(
        input_rank,
        0
    );

    const std::size_t offset =
        output_rank - input_rank;

    for (std::size_t i = 0; i < input_rank; ++i) {
        const std::size_t output_dimension =
            i + offset;

        if (input_shape[i] == 1) {
            input_coordinates[i] = 0;
        } else {
            input_coordinates[i] =
                output_coordinates[output_dimension];
        }
    }

    return ravel_index(
        input_coordinates,
        input_shape
    );
}

// ============================================================
// GENERIC ELEMENTWISE BINARY
// ============================================================

template <typename Operation>
Tensor binary_elementwise(
    const Tensor& a,
    const Tensor& b,
    Operation operation,
    const char* operation_name
) {
    require_float32(a, operation_name);
    require_float32(b, operation_name);

    require_same_dtype(
        a,
        b,
        operation_name
    );

    require_same_device(
        a,
        b,
        operation_name
    );

    const Shape output_shape =
        broadcast_shape(
            a.shape(),
            b.shape()
        );

    Tensor result =
        Tensor::zeros(
            output_shape,
            DType::Float32,
            a.device()
        );

    const float* a_data =
        a.data_as<float>();

    const float* b_data =
        b.data_as<float>();

    float* output =
        result.data_as<float>();

    for (std::size_t i = 0;
         i < result.numel();
         ++i) {

        const std::vector<std::size_t>
            coordinates =
                unravel_index(
                    i,
                    output_shape
                );

        const std::size_t a_index =
            broadcast_input_index(
                coordinates,
                a.shape()
            );

        const std::size_t b_index =
            broadcast_input_index(
                coordinates,
                b.shape()
            );

        output[i] =
            operation(
                a_data[a_index],
                b_data[b_index]
            );
    }

    return result;
}

} // namespace

// ============================================================
// ADD
// ============================================================

Tensor add(
    const Tensor& a,
    const Tensor& b
) {
    Tensor result =
        binary_elementwise(
            a,
            b,
            [](float x, float y) {
                return x + y;
            },
            "add"
        );

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_add_node(
                a,
                b
            )
        );
    }

    return result;
}

// ============================================================
// SUB
// ============================================================

Tensor sub(
    const Tensor& a,
    const Tensor& b
) {
    Tensor result =
        binary_elementwise(
            a,
            b,
            [](float x, float y) {
                return x - y;
            },
            "sub"
        );

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_sub_node(
                a,
                b
            )
        );
    }

    return result;
}

// ============================================================
// MUL
// ============================================================

Tensor mul(
    const Tensor& a,
    const Tensor& b
) {
    Tensor result =
        binary_elementwise(
            a,
            b,
            [](float x, float y) {
                return x * y;
            },
            "mul"
        );

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_mul_node(
                a,
                b
            )
        );
    }

    return result;
}

// ============================================================
// DIV
// ============================================================

Tensor div(
    const Tensor& a,
    const Tensor& b
) {
    Tensor result =
        binary_elementwise(
            a,
            b,
            [](float x, float y) {
                return x / y;
            },
            "div"
        );

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_div_node(
                a,
                b
            )
        );
    }

    return result;
}

// ============================================================
// NEG
// ============================================================

Tensor neg(
    const Tensor& input
) {
    require_float32(
        input,
        "neg"
    );

    Tensor result =
        Tensor::empty(
            input.shape(),
            DType::Float32,
            input.device()
        );

    const float* source =
        input.data_as<float>();

    float* destination =
        result.data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        destination[i] =
            -source[i];
    }

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_neg_node(
                input
            )
        );
    }

    return result;
}

// ============================================================
// SUM
// ============================================================

Tensor sum(
    const Tensor& input
) {
    require_float32(
        input,
        "sum"
    );

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            input.device()
        );

    const float* data =
        input.data_as<float>();

    float total = 0.0f;

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        total += data[i];
    }

    result.data_as<float>()[0] =
        total;

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_sum_node(
                input
            )
        );
    }

    return result;
}


// ============================================================
// MEAN
// ============================================================

Tensor mean(
    const Tensor& input
) {
    require_float32(
        input,
        "mean"
    );

    if (input.numel() == 0) {
        throw std::runtime_error(
            "mean: cannot calculate mean of empty tensor"
        );
    }

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            input.device()
        );

    const float* data =
        input.data_as<float>();

    float total = 0.0f;

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        total += data[i];
    }

    result.data_as<float>()[0] =
        total /
        static_cast<float>(input.numel());

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_mean_node(
                input
            )
        );
    }

    return result;
}


// ============================================================
// MAX
// ============================================================

Tensor max(
    const Tensor& input
) {
    require_float32(
        input,
        "max"
    );

    if (input.numel() == 0) {
        throw std::runtime_error(
            "max: cannot calculate maximum of empty tensor"
        );
    }

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            input.device()
        );

    const float* data =
        input.data_as<float>();

    float value =
        data[0];

    for (std::size_t i = 1;
         i < input.numel();
         ++i) {

        value =
            std::max(
                value,
                data[i]
            );
    }

    result.data_as<float>()[0] =
        value;

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_max_node(
                input
            )
        );
    }

    return result;
}

// ============================================================
// MIN
// ============================================================

Tensor min(
    const Tensor& input
) {
    require_float32(
        input,
        "min"
    );

    if (input.numel() == 0) {
        throw std::runtime_error(
            "min: cannot calculate minimum of empty tensor"
        );
    }

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            input.device()
        );

    const float* data =
        input.data_as<float>();

    float value =
        data[0];

    for (std::size_t i = 1;
         i < input.numel();
         ++i) {

        value =
            std::min(
                value,
                data[i]
            );
    }

    result.data_as<float>()[0] =
        value;

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_min_node(
                input
            )
        );
    }

    return result;
}

// ============================================================
// DOT
// ============================================================

Tensor dot(
    const Tensor& a,
    const Tensor& b
) {
    require_float32(a, "dot");
    require_float32(b, "dot");

    require_same_dtype(
        a,
        b,
        "dot"
    );

    require_same_device(
        a,
        b,
        "dot"
    );

    if (a.ndim() != 1 ||
        b.ndim() != 1) {

        throw std::runtime_error(
            "dot: both tensors must be 1D"
        );
    }

    if (a.shape()[0] != b.shape()[0]) {
        throw std::runtime_error(
            "dot: vector sizes must match"
        );
    }

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            a.device()
        );

    const float* a_data =
        a.data_as<float>();

    const float* b_data =
        b.data_as<float>();

    float value = 0.0f;

    for (std::size_t i = 0;
         i < a.shape()[0];
         ++i) {

        value +=
            a_data[i] *
            b_data[i];
    }

    result.data_as<float>()[0] =
        value;

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_matmul_node(
                a,
                b
            )
        );
    }

    return result;
}

// ============================================================
// TRANSPOSE
// ============================================================

Tensor transpose(
    const Tensor& input
) {
    require_float32(
        input,
        "transpose"
    );

    if (input.ndim() < 2) {
        return input;
    }

    const std::size_t rank =
        input.ndim();

    return transpose(
        input,
        rank - 2,
        rank - 1
    );
}

// ============================================================
// TRANSPOSE DIMENSIONS
// ============================================================

Tensor transpose(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
) {
    require_float32(
        input,
        "transpose"
    );

    const std::size_t rank =
        input.ndim();

    if (dim0 >= rank ||
        dim1 >= rank) {

        throw std::out_of_range(
            "transpose: dimension out of range"
        );
    }

    if (dim0 == dim1) {
        return input;
    }

    std::vector<std::size_t>
        output_dimensions =
            input.shape().dimensions();

    std::swap(
        output_dimensions[dim0],
        output_dimensions[dim1]
    );

    Tensor result =
        Tensor::zeros(
            Shape(output_dimensions),
            DType::Float32,
            input.device()
        );

    const float* source =
        input.data_as<float>();

    float* destination =
        result.data_as<float>();

    for (std::size_t i = 0;
         i < result.numel();
         ++i) {

        const std::vector<std::size_t>
            output_coordinates =
                unravel_index(
                    i,
                    result.shape()
                );

        std::vector<std::size_t>
            input_coordinates(
                rank
            );

        for (std::size_t dimension = 0;
             dimension < rank;
             ++dimension) {

            if (dimension == dim0) {
                input_coordinates[dimension] =
                    output_coordinates[dim1];
            }
            else if (dimension == dim1) {
                input_coordinates[dimension] =
                    output_coordinates[dim0];
            }
            else {
                input_coordinates[dimension] =
                    output_coordinates[dimension];
            }
        }

        const std::size_t input_index =
            ravel_index(
                input_coordinates,
                input.shape()
            );

        destination[i] =
            source[input_index];
    }

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_transpose_node(
                input,
                dim0,
                dim1
            )
        );
    }

    return result;
}

// ============================================================
// MATMUL
//
// Supported:
// 1D x 1D -> scalar
// 2D x 1D -> vector
// 1D x 2D -> vector
// 2D x 2D -> matrix
// ND x ND -> batched matrix multiplication
//
// Batch dimensions use broadcasting.
// ============================================================

Tensor matmul(
    const Tensor& a,
    const Tensor& b
) {
    require_float32(a, "matmul");
    require_float32(b, "matmul");

    require_same_dtype(
        a,
        b,
        "matmul"
    );

    require_same_device(
        a,
        b,
        "matmul"
    );

    if (a.ndim() == 0 ||
        b.ndim() == 0) {

        throw std::runtime_error(
            "matmul: scalar tensors are not supported"
        );
    }

    // --------------------------------------------------------
    // 1D x 1D
    // --------------------------------------------------------

    if (a.ndim() == 1 &&
        b.ndim() == 1) {

        return dot(a, b);
    }

    const std::size_t a_rank =
        a.ndim();

    const std::size_t b_rank =
        b.ndim();

    const std::size_t a_rows =
        a_rank == 1
            ? 1
            : a.shape()[a_rank - 2];

    const std::size_t a_cols =
        a.shape()[a_rank - 1];

    const std::size_t b_rows =
        b_rank == 1
            ? b.shape()[0]
            : b.shape()[b_rank - 2];

    const std::size_t b_cols =
        b_rank == 1
            ? 1
            : b.shape()[b_rank - 1];

    if (a_cols != b_rows) {
        std::ostringstream message;

        message
            << "matmul: incompatible matrix dimensions "
            << a.shape().to_string()
            << " and "
            << b.shape().to_string();

        throw std::runtime_error(
            message.str()
        );
    }

    // --------------------------------------------------------
    // Batch dimensions
    // --------------------------------------------------------

    Shape a_batch;

    Shape b_batch;

    if (a_rank > 2) {
        std::vector<std::size_t>
            dimensions(
                a.shape().dimensions().begin(),
                a.shape().dimensions().end() - 2
            );

        a_batch =
            Shape(dimensions);
    }

    if (b_rank > 2) {
        std::vector<std::size_t>
            dimensions(
                b.shape().dimensions().begin(),
                b.shape().dimensions().end() - 2
            );

        b_batch =
            Shape(dimensions);
    }

    const Shape batch_shape =
        broadcast_shape(
            a_batch,
            b_batch
        );

    // --------------------------------------------------------
    // Output shape
    // --------------------------------------------------------

    std::vector<std::size_t>
        output_dimensions =
            batch_shape.dimensions();

    if (a_rank != 1) {
        output_dimensions.push_back(
            a_rows
        );
    }

    if (b_rank != 1) {
        output_dimensions.push_back(
            b_cols
        );
    }

    Tensor result =
        Tensor::zeros(
            Shape(output_dimensions),
            DType::Float32,
            a.device()
        );

    const float* a_data =
        a.data_as<float>();

    const float* b_data =
        b.data_as<float>();

    float* output =
        result.data_as<float>();

    const std::size_t batch_count =
        batch_shape.numel();

    // --------------------------------------------------------
    // Iterate batches
    // --------------------------------------------------------

    for (std::size_t batch_index = 0;
         batch_index < batch_count;
         ++batch_index) {

        const std::vector<std::size_t>
            batch_coordinates =
                unravel_index(
                    batch_index,
                    batch_shape
                );

        // Map output batch coordinate
        // to A batch coordinate.
        std::vector<std::size_t>
            a_batch_coordinates;

        if (a_batch.ndim() != 0) {
            const std::size_t offset =
                batch_shape.ndim() -
                a_batch.ndim();

            a_batch_coordinates.resize(
                a_batch.ndim()
            );

            for (std::size_t i = 0;
                 i < a_batch.ndim();
                 ++i) {

                if (a_batch[i] == 1) {
                    a_batch_coordinates[i] =
                        0;
                }
                else {
                    a_batch_coordinates[i] =
                        batch_coordinates[
                            i + offset
                        ];
                }
            }
        }

        // Map output batch coordinate
        // to B batch coordinate.
        std::vector<std::size_t>
            b_batch_coordinates;

        if (b_batch.ndim() != 0) {
            const std::size_t offset =
                batch_shape.ndim() -
                b_batch.ndim();

            b_batch_coordinates.resize(
                b_batch.ndim()
            );

            for (std::size_t i = 0;
                 i < b_batch.ndim();
                 ++i) {

                if (b_batch[i] == 1) {
                    b_batch_coordinates[i] =
                        0;
                }
                else {
                    b_batch_coordinates[i] =
                        batch_coordinates[
                            i + offset
                        ];
                }
            }
        }

        // Convert batch coordinates to
        // matrix base offsets.
        std::size_t a_batch_offset = 0;

        if (a_rank > 2) {
            std::vector<std::size_t>
                full_a_coordinates =
                    a_batch_coordinates;

            full_a_coordinates.push_back(0);
            full_a_coordinates.push_back(0);

            a_batch_offset =
                ravel_index(
                    full_a_coordinates,
                    a.shape()
                );
        }

        std::size_t b_batch_offset = 0;

        if (b_rank > 2) {
            std::vector<std::size_t>
                full_b_coordinates =
                    b_batch_coordinates;

            full_b_coordinates.push_back(0);
            full_b_coordinates.push_back(0);

            b_batch_offset =
                ravel_index(
                    full_b_coordinates,
                    b.shape()
                );
        }

        // ----------------------------------------------------
        // Matrix multiplication
        //
        // CPU OPTIMIZATION: I-K-J LOOP ORDER
        //
        // Previous order:
        //
        //     row -> column -> k
        //
        // Optimized order:
        //
        //     row -> k -> column
        //
        // For contiguous row-major matrices this gives:
        //
        //     A[row, k]       : reused across all columns
        //     B[k, column]    : contiguous over column
        //     C[row, column]  : contiguous over column
        //
        // This layout is also a better foundation for future
        // OpenMP parallelization and ARM NEON vectorization.
        // ----------------------------------------------------

        for (std::size_t row = 0;
             row < a_rows;
             ++row) {

            // ------------------------------------------------
            // 2D/ND output row base
            //
            // For vector outputs (2D x 1D), the output is
            // handled separately below because there is no
            // column dimension in the result.
            // ------------------------------------------------

            if (b_rank == 1) {

                float value = 0.0f;

                for (std::size_t k = 0;
                     k < a_cols;
                     ++k) {

                    const std::size_t
                        a_index =
                            a_rank == 1
                                ? a_batch_offset + k
                                : a_batch_offset +
                                  row * a_cols +
                                  k;

                    const std::size_t
                        b_index =
                            b_batch_offset + k;

                    value +=
                        a_data[a_index] *
                        b_data[b_index];
                }

                // ------------------------------------------------
                // The generic output layout may contain
                // broadcasted/multidimensional batch
                // dimensions, so calculate the final index
                // using the existing shape machinery.
                // ------------------------------------------------

                std::vector<std::size_t>
                    output_coordinates =
                        batch_coordinates;

                if (a_rank != 1) {
                    output_coordinates.push_back(
                        row
                    );
                }

                const std::size_t
                    output_index =
                        ravel_index(
                            output_coordinates,
                            result.shape()
                        );

                output[output_index] =
                    value;

                continue;
            }

            // ------------------------------------------------
            // Matrix output.
            //
            // I-K-J:
            //
            //     row
            //       k
            //         column
            //
            // Instead of repeatedly loading/storing C for
            // every k, accumulate directly into the output
            // row. The output row is contiguous.
            // ------------------------------------------------

            std::vector<std::size_t>
                row_output_coordinates =
                    batch_coordinates;

            if (a_rank != 1) {
                row_output_coordinates.push_back(
                    row
                );
            }

            row_output_coordinates.push_back(
                0
            );

            const std::size_t
                output_row_offset =
                    ravel_index(
                        row_output_coordinates,
                        result.shape()
                    );

            float* output_row =
                output +
                output_row_offset;

            // ------------------------------------------------
            // Initialize output row.
            // ------------------------------------------------

            for (std::size_t column = 0;
                 column < b_cols;
                 ++column) {

                output_row[column] =
                    0.0f;
            }

            // ------------------------------------------------
            // I-K-J kernel.
            // ------------------------------------------------

            for (std::size_t k = 0;
                 k < a_cols;
                 ++k) {

                const std::size_t
                    a_index =
                        a_rank == 1
                            ? a_batch_offset + k
                            : a_batch_offset +
                              row * a_cols +
                              k;

                const float
                    a_value =
                        a_data[a_index];

                const std::size_t
                    b_row_offset =
                        b_batch_offset +
                        k * b_cols;

                for (std::size_t column = 0;
                     column < b_cols;
                     ++column) {

                    output_row[column] +=
                        a_value *
                        b_data[
                            b_row_offset +
                            column
                        ];
                }
            }
        }
    }

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (a.requires_grad() ||
        b.requires_grad()) {

        result.set_grad_fn(
            make_matmul_node(
                a,
                b
            )
        );
    }

    return result;
}

} // namespace venla
