#include "venla/nn/linear.hpp"

#include "venla/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// PROPAGATE GRADIENT TO PARENT
// ============================================================

void propagate_to_parent(
    const Tensor& parent,
    const Tensor& gradient
) {
    if (!parent.requires_grad()) {
        return;
    }

    if (gradient.shape() != parent.shape()) {
        throw std::runtime_error(
            "Linear backward: gradient shape mismatch"
        );
    }

    if (gradient.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Linear backward: gradient must be Float32"
        );
    }

    parent.accumulate_grad(
        gradient
    );

    if (parent.grad_state() &&
        parent.grad_state()->grad_fn) {

        parent.grad_state()
            ->grad_fn
            ->backward(
                gradient
            );
    }
}

// ============================================================
// LINEAR AUTOGRAD NODE
//
// Forward:
//
//     y = xW + b
//
// Weight:
//
//     [in_features, out_features]
//
// Input:
//
//     [..., in_features]
//
// Output:
//
//     [..., out_features]
//
// Backward:
//
//     dX = dY W^T
//
//     dW = X^T dY
//
//     db = sum(dY over all leading dimensions)
//
// ============================================================

std::shared_ptr<AutogradNode>
make_linear_node(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    bool use_bias,
    std::size_t in_features,
    std::size_t out_features
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
            use_bias,
            in_features,
            out_features
        ](const Tensor& gradient) mutable {

            // ------------------------------------------------
            // Validate upstream gradient.
            // ------------------------------------------------

            if (gradient.dtype() != DType::Float32) {
                throw std::runtime_error(
                    "Linear backward: "
                    "gradient must be Float32"
                );
            }

            if (gradient.shape().ndim() !=
                input.shape().ndim()) {

                throw std::runtime_error(
                    "Linear backward: "
                    "gradient rank mismatch"
                );
            }

            for (std::size_t dimension = 0;
                 dimension < gradient.ndim();
                 ++dimension) {

                if (
                    gradient.shape()[dimension] !=
                    input.shape()[dimension]
                ) {

                    // Last dimension differs between input
                    // and output, therefore handle it below.
                    if (
                        dimension !=
                        gradient.ndim() - 1
                    ) {

                        throw std::runtime_error(
                            "Linear backward: "
                            "gradient leading dimension mismatch"
                        );
                    }
                }
            }

            if (
                gradient.shape()[
                    gradient.ndim() - 1
                ] != out_features
            ) {

                throw std::runtime_error(
                    "Linear backward: "
                    "gradient output feature dimension mismatch"
                );
            }

            // ------------------------------------------------
            // Number of flattened rows.
            //
            // Input:
            //
            //     [..., in_features]
            //
            // Output:
            //
            //     [..., out_features]
            //
            // Each leading-dimension combination is one row.
            // ------------------------------------------------

            const std::size_t rows =
                input.numel() /
                in_features;

            // ------------------------------------------------
            // Data pointers.
            // ------------------------------------------------

            const float* x =
                input.data_as<float>();

            const float* w =
                weight.data_as<float>();

            const float* dy =
                gradient.data_as<float>();

            // ------------------------------------------------
            // Gradient input.
            //
            // Shape identical to input.
            // ------------------------------------------------

            if (input.requires_grad()) {

                Tensor grad_input =
                    Tensor::zeros(
                        input.shape(),
                        DType::Float32,
                        input.device()
                    );

                float* dx =
                    grad_input.data_as<float>();

                for (std::size_t row = 0;
                     row < rows;
                     ++row) {

                    const float* dy_row =
                        dy +
                        row * out_features;

                    float* dx_row =
                        dx +
                        row * in_features;

                    for (std::size_t in = 0;
                         in < in_features;
                         ++in) {

                        float value = 0.0f;

                        for (std::size_t out = 0;
                             out < out_features;
                             ++out) {

                            value +=
                                dy_row[out] *
                                w[
                                    in *
                                    out_features +
                                    out
                                ];
                        }

                        dx_row[in] =
                            value;
                    }
                }

                propagate_to_parent(
                    input,
                    grad_input
                );
            }

            // ------------------------------------------------
            // Gradient weight.
            //
            // dW =
            //
            //     X^T dY
            //
            // Weight shape:
            //
            //     [in_features, out_features]
            // ------------------------------------------------

            if (weight.requires_grad()) {

                Tensor grad_weight =
                    Tensor::zeros(
                        weight.shape(),
                        DType::Float32,
                        weight.device()
                    );

                float* dw =
                    grad_weight.data_as<float>();

                for (std::size_t row = 0;
                     row < rows;
                     ++row) {

                    const float* x_row =
                        x +
                        row * in_features;

                    const float* dy_row =
                        dy +
                        row * out_features;

                    for (std::size_t in = 0;
                         in < in_features;
                         ++in) {

                        const float x_value =
                            x_row[in];

                        for (std::size_t out = 0;
                             out < out_features;
                             ++out) {

                            dw[
                                in *
                                out_features +
                                out
                            ] +=
                                x_value *
                                dy_row[out];
                        }
                    }
                }

                propagate_to_parent(
                    weight,
                    grad_weight
                );
            }

            // ------------------------------------------------
            // Gradient bias.
            //
            // db = sum(dY over rows)
            //
            // Bias shape:
            //
            //     [out_features]
            // ------------------------------------------------

            if (
                use_bias &&
                bias.requires_grad()
            ) {

                Tensor grad_bias =
                    Tensor::zeros(
                        bias.shape(),
                        DType::Float32,
                        bias.device()
                    );

                float* db =
                    grad_bias.data_as<float>();

                for (std::size_t row = 0;
                     row < rows;
                     ++row) {

                    const float* dy_row =
                        dy +
                        row * out_features;

                    for (std::size_t out = 0;
                         out < out_features;
                         ++out) {

                        db[out] +=
                            dy_row[out];
                    }
                }

                propagate_to_parent(
                    bias,
                    grad_bias
                );
            }
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features,
    bool use_bias
)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(),
      bias_() {

    if (in_features == 0) {
        throw std::invalid_argument(
            "Linear: "
            "in_features must be greater than zero"
        );
    }

    if (out_features == 0) {
        throw std::invalid_argument(
            "Linear: "
            "out_features must be greater than zero"
        );
    }

    weight_ =
        Tensor::empty(
            {
                in_features_,
                out_features_
            },
            DType::Float32,
            Device::cpu()
        );

    if (use_bias_) {

        bias_ =
            Tensor::empty(
                {
                    out_features_
                },
                DType::Float32,
                Device::cpu()
            );
    }
    else {

        bias_ =
            Tensor();
    }

    // --------------------------------------------------------
    // Trainable parameters.
    // --------------------------------------------------------

    weight_.requires_grad_(true);

    if (use_bias_) {
        bias_.requires_grad_(true);
    }

    reset_parameters();
}

// ============================================================
// RESET PARAMETERS
//
// Xavier / Glorot uniform:
//
//     limit = sqrt(6 / (fan_in + fan_out))
//
//     weight ~ U(-limit, +limit)
//
// Bias:
//
//     0
//
// Deterministic seed is retained for reproducible tests.
// ============================================================

void Linear::reset_parameters() {

    if (weight_.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Linear::reset_parameters: "
            "weight must be Float32"
        );
    }

    if (!weight_.device().is_cpu()) {
        throw std::runtime_error(
            "Linear::reset_parameters: "
            "only CPU device is currently supported"
        );
    }

    const float fan_in =
        static_cast<float>(
            in_features_
        );

    const float fan_out =
        static_cast<float>(
            out_features_
        );

    const float limit =
        std::sqrt(
            6.0f /
            (fan_in + fan_out)
        );

    std::mt19937 generator(
        0x56454E4Cu
    );

    std::uniform_real_distribution<float>
        distribution(
            -limit,
            limit
        );

    float* weights =
        weight_.data_as<float>();

    for (std::size_t i = 0;
         i < weight_.numel();
         ++i) {

        weights[i] =
            distribution(
                generator
            );
    }

    if (use_bias_) {

        if (bias_.dtype() != DType::Float32) {
            throw std::runtime_error(
                "Linear::reset_parameters: "
                "bias must be Float32"
            );
        }

        float* biases =
            bias_.data_as<float>();

        for (std::size_t i = 0;
             i < bias_.numel();
             ++i) {

            biases[i] =
                0.0f;
        }
    }
}

// ============================================================
// FORWARD
//
// Input:
//
//     [..., in_features]
//
// Weight:
//
//     [in_features, out_features]
//
// Output:
//
//     [..., out_features]
//
// ============================================================

Tensor Linear::forward(
    const Tensor& input
) const {

    // --------------------------------------------------------
    // Validate input rank.
    // --------------------------------------------------------

    if (input.ndim() == 0) {

        throw std::runtime_error(
            "Linear::forward: "
            "input must have at least 1 dimension"
        );
    }

    // --------------------------------------------------------
    // Validate dtype.
    // --------------------------------------------------------

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "Linear::forward: "
            "currently only Float32 is supported"
        );
    }

    // --------------------------------------------------------
    // Validate device.
    // --------------------------------------------------------

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "Linear::forward: "
            "only CPU device is currently supported"
        );
    }

    // --------------------------------------------------------
    // Validate last dimension.
    // --------------------------------------------------------

    const std::size_t input_features =
        input.shape()[
            input.ndim() - 1
        ];

    if (input_features != in_features_) {

        std::ostringstream message;

        message
            << "Linear::forward: "
            << "input last dimension must be "
            << in_features_
            << ", got "
            << input_features
            << " for input shape "
            << input.shape().to_string();

        throw std::runtime_error(
            message.str()
        );
    }

    // --------------------------------------------------------
    // Output shape.
    //
    // Replace last dimension:
    //
    //     [..., in_features]
    //
    // with:
    //
    //     [..., out_features]
    // --------------------------------------------------------

    std::vector<std::size_t>
        output_dimensions =
            input.shape().dimensions();

    output_dimensions.back() =
        out_features_;

    Tensor result =
        Tensor::zeros(
            Shape(output_dimensions),
            DType::Float32,
            input.device()
        );

    // --------------------------------------------------------
    // Pointers.
    // --------------------------------------------------------

    const float* x =
        input.data_as<float>();

    const float* w =
        weight_.data_as<float>();

    float* y =
        result.data_as<float>();

    const float* b =
        use_bias_
            ? bias_.data_as<float>()
            : nullptr;

    // --------------------------------------------------------
    // Flatten all leading dimensions.
    // --------------------------------------------------------

    const std::size_t rows =
        input.numel() /
        in_features_;

    // --------------------------------------------------------
    // Matrix multiplication:
    //
    //     y = xW + b
    // --------------------------------------------------------

    for (std::size_t row = 0;
         row < rows;
         ++row) {

        const float* x_row =
            x +
            row * in_features_;

        float* y_row =
            y +
            row * out_features_;

        for (std::size_t out = 0;
             out < out_features_;
             ++out) {

            float value =
                0.0f;

            for (std::size_t in = 0;
                 in < in_features_;
                 ++in) {

                value +=
                    x_row[in] *
                    w[
                        in *
                        out_features_ +
                        out
                    ];
            }

            if (b != nullptr) {
                value +=
                    b[out];
            }

            y_row[out] =
                value;
        }
    }

    // --------------------------------------------------------
    // AUTOGRAD
    //
    // The output requires gradients whenever at least one
    // differentiable parent requires gradients.
    // --------------------------------------------------------

    if (
        input.requires_grad() ||
        weight_.requires_grad() ||
        (
            use_bias_ &&
            bias_.requires_grad()
        )
    ) {

        std::shared_ptr<AutogradNode>
            node =
                make_linear_node(
                    input,
                    weight_,
                    bias_,
                    use_bias_,
                    in_features_,
                    out_features_
                );

        result.set_grad_fn(
            node
        );
    }

    return result;
}

// ============================================================
// WEIGHT
// ============================================================

const Tensor&
Linear::weight() const {
    return weight_;
}

Tensor&
Linear::weight() {
    return weight_;
}

// ============================================================
// BIAS
// ============================================================

const Tensor&
Linear::bias() const {
    return bias_;
}

Tensor&
Linear::bias() {
    return bias_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
Linear::in_features() const {
    return in_features_;
}

std::size_t
Linear::out_features() const {
    return out_features_;
}

bool
Linear::has_bias() const {
    return use_bias_;
}

} // namespace venla
