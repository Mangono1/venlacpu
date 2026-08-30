#include "venla/autograd/ops.hpp"

#include "venla/math/operations.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace venla {

namespace {

// ============================================================
// REDUCE BROADCAST GRADIENT
// ============================================================

Tensor reduce_to_shape(
    const Tensor& gradient,
    const Shape& target_shape
) {
    if (gradient.shape() == target_shape) {
        return gradient;
    }

    if (gradient.dtype() != DType::Float32) {
        throw std::runtime_error(
            "reduce_to_shape: only Float32 supported"
        );
    }

    const std::size_t output_rank =
        gradient.ndim();

    const std::size_t target_rank =
        target_shape.ndim();

    if (target_rank > output_rank) {
        throw std::runtime_error(
            "reduce_to_shape: target rank exceeds gradient rank"
        );
    }

    for (std::size_t i = 0; i < target_rank; ++i) {
        const std::size_t out_dim =
            output_rank - target_rank + i;

        const std::size_t target_dim =
            target_shape[i];

        const std::size_t output_dim =
            gradient.shape()[out_dim];

        if (target_dim != 1 &&
            target_dim != output_dim) {
            throw std::runtime_error(
                "reduce_to_shape: incompatible broadcast shape"
            );
        }
    }

    Tensor result =
        Tensor::zeros(
            target_shape,
            DType::Float32,
            gradient.device()
        );

    const float* source =
        gradient.data_as<float>();

    float* destination =
        result.data_as<float>();

    for (std::size_t linear = 0;
         linear < gradient.numel();
         ++linear) {

        std::size_t index = linear;

        std::vector<std::size_t>
            output_coordinates(
                output_rank,
                0
            );

        for (std::size_t i = output_rank;
             i > 0;
             --i) {

            const std::size_t dimension =
                i - 1;

            output_coordinates[dimension] =
                index %
                gradient.shape()[dimension];

            index /=
                gradient.shape()[dimension];
        }

        if (target_rank == 0) {

            destination[0] +=
                source[linear];

            continue;
        }

        const std::size_t offset =
            output_rank -
            target_rank;

        std::vector<std::size_t>
            target_coordinates(
                target_rank,
                0
            );

        for (std::size_t i = 0;
             i < target_rank;
             ++i) {

            const std::size_t
                output_dimension =
                    i + offset;

            if (target_shape[i] == 1) {

                target_coordinates[i] =
                    0;
            }
            else {

                target_coordinates[i] =
                    output_coordinates[
                        output_dimension
                    ];
            }
        }

        std::size_t target_index = 0;

        for (std::size_t i = 0;
             i < target_rank;
             ++i) {

            target_index =
                target_index *
                target_shape[i] +
                target_coordinates[i];
        }

        destination[target_index] +=
            source[linear];
    }

    return result;
}

// ============================================================
// BACKPROPAGATE TO PARENT
// ============================================================

void propagate(
    const Tensor& parent,
    const Tensor& gradient
) {
    if (!parent.requires_grad()) {
        return;
    }

    Tensor reduced =
        reduce_to_shape(
            gradient,
            parent.shape()
        );

    parent.accumulate_grad(
        reduced
    );

    if (parent.grad_state()->grad_fn) {

        parent.grad_state()
            ->grad_fn
            ->backward(
                reduced
            );
    }
}

} // namespace

// ============================================================
// ADD
// ============================================================

std::shared_ptr<AutogradNode>
make_add_node(
    const Tensor& a,
    const Tensor& b
) {
    if (!a.requires_grad() &&
        !b.requires_grad()) {

        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{a, b},

        [a, b](const Tensor& gradient) mutable {

            if (a.requires_grad()) {
                propagate(
                    a,
                    gradient
                );
            }

            if (b.requires_grad()) {
                propagate(
                    b,
                    gradient
                );
            }
        }
    );
}

// ============================================================
// SUB
// ============================================================

std::shared_ptr<AutogradNode>
make_sub_node(
    const Tensor& a,
    const Tensor& b
) {
    if (!a.requires_grad() &&
        !b.requires_grad()) {

        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{a, b},

        [a, b](const Tensor& gradient) mutable {

            if (a.requires_grad()) {
                propagate(
                    a,
                    gradient
                );
            }

            if (b.requires_grad()) {

                Tensor negative_gradient =
                    neg(
                        gradient
                    );

                propagate(
                    b,
                    negative_gradient
                );
            }
        }
    );
}

// ============================================================
// MUL
// ============================================================

std::shared_ptr<AutogradNode>
make_mul_node(
    const Tensor& a,
    const Tensor& b
) {
    if (!a.requires_grad() &&
        !b.requires_grad()) {

        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{a, b},

        [a, b](const Tensor& gradient) mutable {

            if (a.requires_grad()) {

                Tensor grad_a =
                    mul(
                        gradient,
                        b
                    );

                propagate(
                    a,
                    grad_a
                );
            }

            if (b.requires_grad()) {

                Tensor grad_b =
                    mul(
                        gradient,
                        a
                    );

                propagate(
                    b,
                    grad_b
                );
            }
        }
    );
}

// ============================================================
// DIV
// ============================================================

std::shared_ptr<AutogradNode>
make_div_node(
    const Tensor& a,
    const Tensor& b
) {
    if (!a.requires_grad() &&
        !b.requires_grad()) {

        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{a, b},

        [a, b](const Tensor& gradient) mutable {

            if (a.requires_grad()) {

                Tensor grad_a =
                    div(
                        gradient,
                        b
                    );

                propagate(
                    a,
                    grad_a
                );
            }

            if (b.requires_grad()) {

                Tensor numerator =
                    mul(
                        gradient,
                        a
                    );

                Tensor denominator =
                    mul(
                        b,
                        b
                    );

                Tensor grad_b =
                    div(
                        numerator,
                        denominator
                    );

                grad_b =
                    neg(
                        grad_b
                    );

                propagate(
                    b,
                    grad_b
                );
            }
        }
    );
}

// ============================================================
// NEG
// ============================================================

std::shared_ptr<AutogradNode>
make_neg_node(
    const Tensor& input
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input](const Tensor& gradient) mutable {

            Tensor negative_gradient =
                neg(
                    gradient
                );

            propagate(
                input,
                negative_gradient
            );
        }
    );
}

// ============================================================
// SUM
// ============================================================

std::shared_ptr<AutogradNode>
make_sum_node(
    const Tensor& input
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input](const Tensor& gradient) mutable {

            Tensor expanded =
                Tensor::ones(
                    input.shape(),
                    DType::Float32,
                    input.device()
                );

            Tensor grad_input =
                mul(
                    expanded,
                    gradient
                );

            propagate(
                input,
                grad_input
            );
        }
    );
}

// ============================================================
// MEAN
// ============================================================

std::shared_ptr<AutogradNode>
make_mean_node(
    const Tensor& input
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input](const Tensor& gradient) mutable {

            Tensor expanded =
                Tensor::ones(
                    input.shape(),
                    DType::Float32,
                    input.device()
                );

            Tensor grad_input =
                mul(
                    expanded,
                    gradient
                );

            Tensor divisor =
                Tensor::zeros(
                    Shape{},
                    DType::Float32,
                    input.device()
                );

            divisor.data_as<float>()[0] =
                static_cast<float>(
                    input.numel()
                );

            Tensor normalized =
                div(
                    grad_input,
                    divisor
                );

            propagate(
                input,
                normalized
            );
        }
    );
}

// ============================================================
// MAX
//
// y = max(x)
//
// Gradient:
//
// d(max(x))/dx_i = 1
//                  jika x_i adalah maximum
//
// Jika terdapat beberapa maximum yang sama,
// gradient dibagi rata:
//
// gradient_i = upstream / jumlah_maximum
//
// Ini membuat total gradient tetap sama dengan upstream.
// ============================================================

std::shared_ptr<AutogradNode>
make_max_node(
    const Tensor& input
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input](const Tensor& gradient) mutable {

            if (gradient.numel() != 1) {
                throw std::runtime_error(
                    "max backward: expected scalar gradient"
                );
            }

            const float* input_data =
                input.data_as<float>();

            float maximum =
                input_data[0];

            for (std::size_t i = 1;
                 i < input.numel();
                 ++i) {

                if (input_data[i] > maximum) {
                    maximum =
                        input_data[i];
                }
            }

            std::size_t count = 0;

            for (std::size_t i = 0;
                 i < input.numel();
                 ++i) {

                if (input_data[i] == maximum) {
                    ++count;
                }
            }

            Tensor grad_input =
                Tensor::zeros(
                    input.shape(),
                    DType::Float32,
                    input.device()
                );

            float* grad_data =
                grad_input.data_as<float>();

            const float upstream =
                gradient.data_as<float>()[0];

            const float share =
                upstream /
                static_cast<float>(count);

            for (std::size_t i = 0;
                 i < input.numel();
                 ++i) {

                if (input_data[i] == maximum) {
                    grad_data[i] = share;
                }
            }

            propagate(
                input,
                grad_input
            );
        }
    );
}

// ============================================================
// MIN
//
// y = min(x)
//
// Gradient:
//
// d(min(x))/dx_i = 1
//                  jika x_i adalah minimum
//
// Jika terdapat beberapa minimum yang sama,
// gradient dibagi rata.
// ============================================================

std::shared_ptr<AutogradNode>
make_min_node(
    const Tensor& input
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input](const Tensor& gradient) mutable {

            if (gradient.numel() != 1) {
                throw std::runtime_error(
                    "min backward: expected scalar gradient"
                );
            }

            const float* input_data =
                input.data_as<float>();

            float minimum =
                input_data[0];

            for (std::size_t i = 1;
                 i < input.numel();
                 ++i) {

                if (input_data[i] < minimum) {
                    minimum =
                        input_data[i];
                }
            }

            std::size_t count = 0;

            for (std::size_t i = 0;
                 i < input.numel();
                 ++i) {

                if (input_data[i] == minimum) {
                    ++count;
                }
            }

            Tensor grad_input =
                Tensor::zeros(
                    input.shape(),
                    DType::Float32,
                    input.device()
                );

            float* grad_data =
                grad_input.data_as<float>();

            const float upstream =
                gradient.data_as<float>()[0];

            const float share =
                upstream /
                static_cast<float>(count);

            for (std::size_t i = 0;
                 i < input.numel();
                 ++i) {

                if (input_data[i] == minimum) {
                    grad_data[i] = share;
                }
            }

            propagate(
                input,
                grad_input
            );
        }
    );
}

// ============================================================
// GENERIC BACKWARD HELPER
// ============================================================

void backward_tensor(
    const Tensor& tensor,
    const Tensor& gradient
) {
    if (!tensor.requires_grad()) {
        return;
    }

    if (gradient.shape() != tensor.shape()) {

        Tensor reduced =
            reduce_to_shape(
                gradient,
                tensor.shape()
            );

        tensor.accumulate_grad(
            reduced
        );

        if (tensor.grad_state()->grad_fn) {

            tensor.grad_state()
                ->grad_fn
                ->backward(
                    reduced
                );
        }

        return;
    }

    tensor.accumulate_grad(
        gradient
    );

    if (tensor.grad_state()->grad_fn) {

        tensor.grad_state()
            ->grad_fn
            ->backward(
                gradient
            );
    }
}

} // namespace venla

/*
 * ============================================================
 * MATMUL AUTOGRAD
 * ============================================================
 *
 * Supported backward paths:
 *
 *   1D x 1D -> scalar
 *   2D x 1D -> vector
 *   1D x 2D -> vector
 *   2D x 2D -> matrix
 *
 * Batched matmul will be handled in a later stage after the
 * fundamental matrix gradient path is verified.
 *
 * ============================================================
 */

namespace venla {

namespace {

Shape autograd_broadcast_shape(
    const Shape& a,
    const Shape& b
) {
    const std::size_t rank =
        std::max(a.ndim(), b.ndim());

    std::vector<std::size_t> dims(rank, 1);

    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t ai =
            i < a.ndim()
                ? a[a.ndim() - 1 - i]
                : 1;

        const std::size_t bi =
            i < b.ndim()
                ? b[b.ndim() - 1 - i]
                : 1;

        if (ai != bi && ai != 1 && bi != 1) {
            throw std::runtime_error(
                "matmul backward: incompatible batch dimensions"
            );
        }

        dims[rank - 1 - i] =
            ai > bi ? ai : bi;
    }

    return Shape(dims);
}

std::vector<std::size_t> autograd_unravel(
    std::size_t index,
    const Shape& shape
) {
    std::vector<std::size_t> coordinates(
        shape.ndim(),
        0
    );

    for (std::size_t i = shape.ndim(); i > 0; --i) {
        const std::size_t dim = i - 1;

        coordinates[dim] =
            index % shape[dim];

        index /= shape[dim];
    }

    return coordinates;
}

std::size_t autograd_ravel(
    const std::vector<std::size_t>& coordinates,
    const Shape& shape
) {
    std::size_t index = 0;

    for (std::size_t i = 0;
         i < coordinates.size();
         ++i) {

        index =
            index * shape[i] +
            coordinates[i];
    }

    return index;
}

Tensor matmul_outer_product(
    const Tensor& a,
    const Tensor& b
) {
    if (a.ndim() != 1 ||
        b.ndim() != 1) {

        throw std::runtime_error(
            "matmul_outer_product: "
            "both tensors must be 1D"
        );
    }

    Tensor result =
        Tensor::zeros(
            {a.shape()[0], b.shape()[0]},
            DType::Float32,
            a.device()
        );

    const float* av =
        a.data_as<float>();

    const float* bv =
        b.data_as<float>();

    float* rv =
        result.data_as<float>();

    for (std::size_t i = 0;
         i < a.shape()[0];
         ++i) {

        for (std::size_t j = 0;
             j < b.shape()[0];
             ++j) {

            rv[
                i * b.shape()[0] + j
            ] =
                av[i] * bv[j];
        }
    }

    return result;
}

} // namespace

std::shared_ptr<AutogradNode>
make_matmul_node(
    const Tensor& a,
    const Tensor& b
) {
    if (!a.requires_grad() && !b.requires_grad())
        return nullptr;

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{a, b},

        [a, b](const Tensor& gradient) mutable {

            if (a.ndim() == 1 && b.ndim() == 1) {
                const float upstream =
                    gradient.data_as<float>()[0];

                if (a.requires_grad()) {
                    Tensor ga = Tensor::zeros(
                        a.shape(), DType::Float32, a.device()
                    );

                    const float* bv = b.data_as<float>();
                    float* av = ga.data_as<float>();

                    for (std::size_t i = 0; i < a.numel(); ++i)
                        av[i] = upstream * bv[i];

                    propagate(a, ga);
                }

                if (b.requires_grad()) {
                    Tensor gb = Tensor::zeros(
                        b.shape(), DType::Float32, b.device()
                    );

                    const float* av = a.data_as<float>();
                    float* bv = gb.data_as<float>();

                    for (std::size_t i = 0; i < b.numel(); ++i)
                        bv[i] = upstream * av[i];

                    propagate(b, gb);
                }

                return;
            }

            /*
             * Batched/general matrix gradient.
             *
             * Handles:
             *
             *   2D x 2D
             *   3D x 2D
             *   2D x 3D
             *   3D x 3D
             *   4D x 2D
             *   2D x 4D
             *   4D x 3D
             *   3D x 4D
             *   4D x 4D
             *
             * Batch dimensions follow normal broadcasting.
             */

            if (a.ndim() < 2 || b.ndim() < 2) {

                if (a.ndim() == 2 && b.ndim() == 1) {
                    if (a.requires_grad()) {
                        Tensor ga =
                            matmul_outer_product(
                                gradient, b
                            );
                        propagate(a, ga);
                    }

                    if (b.requires_grad()) {
                        Tensor ga =
                            transpose(a);

                        Tensor gb =
                            matmul(
                                ga,
                                gradient
                            );

                        propagate(b, gb);
                    }

                    return;
                }

                if (a.ndim() == 1 && b.ndim() == 2) {
                    if (a.requires_grad()) {
                        Tensor bt = transpose(b);

                        Tensor ga =
                            matmul(
                                gradient,
                                bt
                            );

                        propagate(a, ga);
                    }

                    if (b.requires_grad()) {
                        Tensor gb =
                            matmul_outer_product(
                                a,
                                gradient
                            );

                        propagate(b, gb);
                    }

                    return;
                }

                throw std::runtime_error(
                    "matmul backward: unsupported rank combination"
                );
            }

            const std::size_t ar = a.ndim();
            const std::size_t br = b.ndim();

            const std::size_t m =
                a.shape()[ar - 2];

            const std::size_t k =
                a.shape()[ar - 1];

            const std::size_t n =
                b.shape()[br - 1];

            if (b.shape()[br - 2] != k) {
                throw std::runtime_error(
                    "matmul backward: incompatible matrix dimensions"
                );
            }

            Shape a_batch;
            Shape b_batch;

            if (ar > 2) {
                std::vector<std::size_t> d(
                    a.shape().dimensions().begin(),
                    a.shape().dimensions().end() - 2
                );
                a_batch = Shape(d);
            }

            if (br > 2) {
                std::vector<std::size_t> d(
                    b.shape().dimensions().begin(),
                    b.shape().dimensions().end() - 2
                );
                b_batch = Shape(d);
            }

            Shape batch =
                autograd_broadcast_shape(a_batch, b_batch);

            std::vector<std::size_t> a_dims =
                a.shape().dimensions();

            std::vector<std::size_t> b_dims =
                b.shape().dimensions();

            Tensor ga =
                a.requires_grad()
                    ? Tensor::zeros(
                        a.shape(),
                        DType::Float32,
                        a.device()
                    )
                    : Tensor();

            Tensor gb =
                b.requires_grad()
                    ? Tensor::zeros(
                        b.shape(),
                        DType::Float32,
                        b.device()
                    )
                    : Tensor();

            const float* av = a.data_as<float>();
            const float* bv = b.data_as<float>();
            const float* gv = gradient.data_as<float>();

            float* gav =
                a.requires_grad()
                    ? ga.data_as<float>()
                    : nullptr;

            float* gbv =
                b.requires_grad()
                    ? gb.data_as<float>()
                    : nullptr;

            const std::size_t batch_count =
                batch.numel();

            for (std::size_t bi = 0;
                 bi < batch_count;
                 ++bi) {

                std::vector<std::size_t> bc =
                    autograd_unravel(bi, batch);

                std::vector<std::size_t> ac;
                std::vector<std::size_t> bc2;

                if (a_batch.ndim() != 0) {
                    std::size_t offset =
                        batch.ndim() - a_batch.ndim();

                    ac.resize(a_batch.ndim());

                    for (std::size_t i = 0;
                         i < a_batch.ndim();
                         ++i) {

                        ac[i] =
                            a_batch[i] == 1
                                ? 0
                                : bc[i + offset];
                    }
                }

                if (b_batch.ndim() != 0) {
                    std::size_t offset =
                        batch.ndim() - b_batch.ndim();

                    bc2.resize(b_batch.ndim());

                    for (std::size_t i = 0;
                         i < b_batch.ndim();
                         ++i) {

                        bc2[i] =
                            b_batch[i] == 1
                                ? 0
                                : bc[i + offset];
                    }
                }

                std::size_t ao = 0;

                if (ar > 2) {
                    std::vector<std::size_t> full = ac;
                    full.push_back(0);
                    full.push_back(0);

                    ao = autograd_ravel(full, a.shape());
                }

                std::size_t bo = 0;

                if (br > 2) {
                    std::vector<std::size_t> full = bc2;
                    full.push_back(0);
                    full.push_back(0);

                    bo = autograd_ravel(full, b.shape());
                }

                std::size_t go = 0;

                {
                    std::vector<std::size_t> full = bc;

                    if (ar != 1)
                        full.push_back(0);

                    if (br != 1)
                        full.push_back(0);

                    go =
                        autograd_ravel(
                            full,
                            gradient.shape()
                        );
                }

                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {

                        const float upstream =
                            gv[
                                go +
                                i * n +
                                j
                            ];

                        if (a.requires_grad()) {
                            for (std::size_t x = 0;
                                 x < k;
                                 ++x) {

                                gav[
                                    ao +
                                    i * k +
                                    x
                                ] +=
                                    upstream *
                                    bv[
                                        bo +
                                        x * n +
                                        j
                                    ];
                            }
                        }

                        if (b.requires_grad()) {
                            for (std::size_t x = 0;
                                 x < k;
                                 ++x) {

                                gbv[
                                    bo +
                                    x * n +
                                    j
                                ] +=
                                    upstream *
                                    av[
                                        ao +
                                        i * k +
                                        x
                                    ];
                            }
                        }
                    }
                }
            }

            if (a.requires_grad())
                propagate(a, ga);

            if (b.requires_grad())
                propagate(b, gb);
        }
    );
}


/*
 * ============================================================
 * TRANSPOSE AUTOGRAD
 * ============================================================
 *
 * Forward:
 *
 *     y = transpose(x, dim0, dim1)
 *
 * Backward:
 *
 *     dx = transpose(dy, dim0, dim1)
 *
 * Karena transpose adalah operasi permutasi dimensi,
 * inverse permutation-nya adalah transpose dengan pasangan
 * dimensi yang sama.
 *
 * Contoh:
 *
 *     x : [2, 3]
 *     y : [3, 2]
 *
 *     dy : [3, 2]
 *     dx : [2, 3]
 *
 * Untuk dimensi yang berbeda:
 *
 *     transpose(x, 0, 2)
 *
 * gradient dikembalikan dengan:
 *
 *     transpose(gradient, 0, 2)
 *
 * ============================================================
 */

std::shared_ptr<AutogradNode>
make_transpose_node(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    if (dim0 == dim1) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input, dim0, dim1](
            const Tensor& gradient
        ) mutable {

            if (gradient.shape() !=
                input.shape()) {

                // Gradient dari transpose memiliki
                // shape hasil transpose.
                //
                // Jadi validasi dilakukan berdasarkan
                // expected output shape.

                std::vector<std::size_t>
                    expected_dimensions =
                        input.shape().dimensions();

                std::swap(
                    expected_dimensions[dim0],
                    expected_dimensions[dim1]
                );

                Shape expected_shape(
                    expected_dimensions
                );

                if (gradient.shape() !=
                    expected_shape) {

                    throw std::runtime_error(
                        "transpose backward: "
                        "gradient shape mismatch"
                    );
                }
            }

            Tensor grad_input =
                transpose(
                    gradient,
                    dim0,
                    dim1
                );

            propagate(
                input,
                grad_input
            );
        }
    );
}

} // namespace venla
