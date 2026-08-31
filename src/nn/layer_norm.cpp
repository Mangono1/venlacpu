#include "venla/nn/layer_norm.hpp"

#include "venla/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// AUTOGRAD NODE
//
// y = gamma * x_hat + beta
//
// where
//
// x_hat = (x - mean) / sqrt(var + eps)
//
// Backward:
//
// dx = gamma * inv_std / N *
//      (N * dy
//       - sum(dy)
//       - x_hat * sum(dy * x_hat))
//
// dgamma = sum(dy * x_hat)
//
// dbeta = sum(dy)
// ============================================================

std::shared_ptr<AutogradNode> make_layer_norm_node(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    std::size_t normalized_shape,
    float eps
) {

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{
            input,
            weight,
            bias
        },

        [
            input,
            weight,
            bias,
            normalized_shape,
            eps
        ](const Tensor& gradient) {

            if (gradient.dtype() != DType::Float32) {
                throw std::runtime_error(
                    "LayerNorm backward: gradient must be Float32"
                );
            }

            if (gradient.shape() != input.shape()) {
                throw std::runtime_error(
                    "LayerNorm backward: gradient shape mismatch"
                );
            }

            const std::size_t total = input.numel();

            if (normalized_shape == 0 ||
                total % normalized_shape != 0) {

                throw std::runtime_error(
                    "LayerNorm backward: invalid normalized shape"
                );
            }

            const std::size_t rows =
                total / normalized_shape;

            const float* x =
                input.data_as<float>();

            const float* w =
                weight.data_as<float>();

            const float* dy =
                gradient.data_as<float>();

            // ------------------------------------------------
            // Allocate gradients only when required.
            // ------------------------------------------------

            Tensor dx;

            if (input.requires_grad()) {
                dx =
                    Tensor::zeros(
                        input.shape(),
                        DType::Float32,
                        input.device()
                    );
            }

            Tensor dw;

            if (weight.requires_grad()) {
                dw =
                    Tensor::zeros(
                        weight.shape(),
                        DType::Float32,
                        weight.device()
                    );
            }

            Tensor db;

            if (bias.requires_grad()) {
                db =
                    Tensor::zeros(
                        bias.shape(),
                        DType::Float32,
                        bias.device()
                    );
            }

            float* dx_data =
                input.requires_grad()
                    ? dx.data_as<float>()
                    : nullptr;

            float* dw_data =
                weight.requires_grad()
                    ? dw.data_as<float>()
                    : nullptr;

            float* db_data =
                bias.requires_grad()
                    ? db.data_as<float>()
                    : nullptr;

            // ------------------------------------------------
            // Process every independent normalization row.
            // ------------------------------------------------

            for (std::size_t row = 0;
                 row < rows;
                 ++row) {

                const std::size_t offset =
                    row * normalized_shape;

                // --------------------------------------------
                // Mean
                // --------------------------------------------

                double mean =
                    0.0;

                for (std::size_t i = 0;
                     i < normalized_shape;
                     ++i) {

                    mean +=
                        static_cast<double>(
                            x[offset + i]
                        );
                }

                mean /=
                    static_cast<double>(
                        normalized_shape
                    );

                // --------------------------------------------
                // Variance
                // --------------------------------------------

                double variance =
                    0.0;

                for (std::size_t i = 0;
                     i < normalized_shape;
                     ++i) {

                    const double difference =
                        static_cast<double>(
                            x[offset + i]
                        ) -
                        mean;

                    variance +=
                        difference *
                        difference;
                }

                variance /=
                    static_cast<double>(
                        normalized_shape
                    );

                const double inv_std =
                    1.0 /
                    std::sqrt(
                        variance +
                        static_cast<double>(eps)
                    );

                // --------------------------------------------
                // Statistics required for input gradient.
                // --------------------------------------------

                double sum_dy =
                    0.0;

                double sum_dy_xhat =
                    0.0;

                for (std::size_t i = 0;
                     i < normalized_shape;
                     ++i) {

                    const double xhat =
                        (
                            static_cast<double>(
                                x[offset + i]
                            ) -
                            mean
                        ) *
                        inv_std;

                    const double dy_value =
                        static_cast<double>(
                            dy[offset + i]
                        );

                    sum_dy +=
                        dy_value;

                    sum_dy_xhat +=
                        dy_value *
                        xhat;

                    if (weight.requires_grad()) {

                        dw_data[i] +=
                            static_cast<float>(
                                dy_value *
                                xhat
                            );
                    }

                    if (bias.requires_grad()) {

                        db_data[i] +=
                            static_cast<float>(
                                dy_value
                            );
                    }
                }

                // --------------------------------------------
                // Input gradient
                // --------------------------------------------

                if (input.requires_grad()) {

                    const double n =
                        static_cast<double>(
                            normalized_shape
                        );

                    for (std::size_t i = 0;
                         i < normalized_shape;
                         ++i) {

                        const double xhat =
                            (
                                static_cast<double>(
                                    x[offset + i]
                                ) -
                                mean
                            ) *
                            inv_std;

                        const double dy_value =
                            static_cast<double>(
                                dy[offset + i]
                            );

                        const double value =
                            static_cast<double>(
                                w[i]
                            ) *
                            inv_std /
                            n *
                            (
                                n * dy_value -
                                sum_dy -
                                xhat *
                                sum_dy_xhat
                            );

                        dx_data[offset + i] =
                            static_cast<float>(
                                value
                            );
                    }
                }
            }

            // ------------------------------------------------
            // Accumulate gradients.
            // ------------------------------------------------

            if (input.requires_grad()) {

                input.accumulate_grad(dx);

                if (input.grad_state()->grad_fn) {

                    input.grad_state()
                        ->grad_fn
                        ->backward(dx);
                }
            }

            if (weight.requires_grad()) {

                weight.accumulate_grad(dw);

                if (weight.grad_state()->grad_fn) {

                    weight.grad_state()
                        ->grad_fn
                        ->backward(dw);
                }
            }

            if (bias.requires_grad()) {

                bias.accumulate_grad(db);

                if (bias.grad_state()->grad_fn) {

                    bias.grad_state()
                        ->grad_fn
                        ->backward(db);
                }
            }
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

LayerNorm::LayerNorm(
    std::size_t normalized_shape,
    float eps
)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(),
      bias_() {

    if (normalized_shape == 0) {

        throw std::invalid_argument(
            "LayerNorm: normalized_shape "
            "must be greater than zero"
        );
    }

    if (!(eps > 0.0f)) {

        throw std::invalid_argument(
            "LayerNorm: eps must be greater than zero"
        );
    }

    weight_ =
        Tensor::ones(
            {normalized_shape_},
            DType::Float32,
            Device::cpu()
        );

    bias_ =
        Tensor::zeros(
            {normalized_shape_},
            DType::Float32,
            Device::cpu()
        );

    weight_.requires_grad_(true);
    bias_.requires_grad_(true);

    reset_parameters();
}

// ============================================================
// RESET PARAMETERS
//
// LayerNorm standard initialization:
//
//   gamma = 1
//   beta  = 0
// ============================================================

void LayerNorm::reset_parameters() {

    if (weight_.dtype() != DType::Float32) {

        throw std::runtime_error(
            "LayerNorm::reset_parameters: "
            "weight must be Float32"
        );
    }

    if (bias_.dtype() != DType::Float32) {

        throw std::runtime_error(
            "LayerNorm::reset_parameters: "
            "bias must be Float32"
        );
    }

    float* weight =
        weight_.data_as<float>();

    float* bias =
        bias_.data_as<float>();

    for (std::size_t i = 0;
         i < normalized_shape_;
         ++i) {

        weight[i] = 1.0f;
        bias[i] = 0.0f;
    }
}

// ============================================================
// FORWARD
// ============================================================

Tensor LayerNorm::forward(
    const Tensor& input
) const {

    // --------------------------------------------------------
    // LayerNorm VENLACPU contract:
    //
    // Input must have at least 2 dimensions.
    //
    // Supported examples:
    //
    //   [batch, features]
    //   [batch, seq, features]
    //   [d1, d2, ..., features]
    //
    // A 1D tensor [features] is intentionally rejected.
    // --------------------------------------------------------

    if (input.ndim() < 2) {

        throw std::runtime_error(
            "LayerNorm::forward: "
            "input must have at least 2 dimensions"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "LayerNorm::forward: "
            "input must be Float32"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "LayerNorm::forward: "
            "only CPU device is currently supported"
        );
    }

    if (weight_.dtype() != DType::Float32 ||
        bias_.dtype() != DType::Float32) {

        throw std::runtime_error(
            "LayerNorm::forward: "
            "parameters must be Float32"
        );
    }

    const std::size_t features =
        input.shape()[input.ndim() - 1];

    if (features != normalized_shape_) {

        throw std::runtime_error(
            "LayerNorm::forward: "
            "input last dimension does not match "
            "normalized_shape"
        );
    }

    const std::size_t rows =
        input.numel() /
        normalized_shape_;

    Tensor result =
        Tensor::zeros(
            input.shape(),
            DType::Float32,
            input.device()
        );

    const float* x =
        input.data_as<float>();

    const float* weight =
        weight_.data_as<float>();

    const float* bias =
        bias_.data_as<float>();

    float* output =
        result.data_as<float>();

    // --------------------------------------------------------
    // Normalize each row independently.
    // --------------------------------------------------------

    for (std::size_t row = 0;
         row < rows;
         ++row) {

        const std::size_t offset =
            row * normalized_shape_;

        // ----------------------------------------------------
        // Mean
        // ----------------------------------------------------

        double mean =
            0.0;

        for (std::size_t i = 0;
             i < normalized_shape_;
             ++i) {

            mean +=
                static_cast<double>(
                    x[offset + i]
                );
        }

        mean /=
            static_cast<double>(
                normalized_shape_
            );

        // ----------------------------------------------------
        // Variance
        // ----------------------------------------------------

        double variance =
            0.0;

        for (std::size_t i = 0;
             i < normalized_shape_;
             ++i) {

            const double difference =
                static_cast<double>(
                    x[offset + i]
                ) -
                mean;

            variance +=
                difference *
                difference;
        }

        variance /=
            static_cast<double>(
                normalized_shape_
            );

        const double inv_std =
            1.0 /
            std::sqrt(
                variance +
                static_cast<double>(eps_)
            );

        // ----------------------------------------------------
        // Normalize + affine transform.
        // ----------------------------------------------------

        for (std::size_t i = 0;
             i < normalized_shape_;
             ++i) {

            const double normalized =
                (
                    static_cast<double>(
                        x[offset + i]
                    ) -
                    mean
                ) *
                inv_std;

            output[offset + i] =
                static_cast<float>(
                    normalized *
                    static_cast<double>(
                        weight[i]
                    ) +
                    static_cast<double>(
                        bias[i]
                    )
                );
        }
    }

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad() ||
        weight_.requires_grad() ||
        bias_.requires_grad()) {

        result.set_grad_fn(
            make_layer_norm_node(
                input,
                weight_,
                bias_,
                normalized_shape_,
                eps_
            )
        );
    }

    return result;
}

// ============================================================
// WEIGHT
// ============================================================

const Tensor& LayerNorm::weight() const {
    return weight_;
}

Tensor& LayerNorm::weight() {
    return weight_;
}

// ============================================================
// BIAS
// ============================================================

const Tensor& LayerNorm::bias() const {
    return bias_;
}

Tensor& LayerNorm::bias() {
    return bias_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t LayerNorm::normalized_shape() const {
    return normalized_shape_;
}

float LayerNorm::eps() const {
    return eps_;
}

} // namespace venla
