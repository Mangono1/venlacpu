#include "venla/nn/feed_forward.hpp"

#include "venla/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// GELU
//
// Exact Gaussian Error Linear Unit:
//
//   GELU(x) = 0.5 * x *
//             (1 + erf(x / sqrt(2)))
//
// Derivative:
//
//   GELU'(x) =
//       0.5 * (1 + erf(x / sqrt(2)))
//       +
//       x * exp(-x² / 2) / sqrt(2*pi)
//
// ============================================================

float gelu_value(
    float x
) {
    const double value =
        static_cast<double>(x);

    const double result =
        0.5 *
        value *
        (
            1.0 +
            std::erf(
                value /
                std::sqrt(2.0)
            )
        );

    return static_cast<float>(
        result
    );
}

float gelu_derivative(
    float x
) {
    const double value =
        static_cast<double>(x);

    const double erf_part =
        0.5 *
        (
            1.0 +
            std::erf(
                value /
                std::sqrt(2.0)
            )
        );

    const double gaussian =
        std::exp(
            -0.5 *
            value *
            value
        ) /
        std::sqrt(
            2.0 *
            3.14159265358979323846
        );

    const double result =
        erf_part +
        value * gaussian;

    return static_cast<float>(
        result
    );
}

// ============================================================
// VALIDATION
// ============================================================

void validate_input(
    const Tensor& input
) {
    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "FeedForward::forward: "
            "input must be 2D or 3D"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "FeedForward::forward: "
            "only Float32 is currently supported"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "FeedForward::forward: "
            "only CPU device is currently supported"
        );
    }
}

// ============================================================
// LINEAR FORWARD
//
// x:
//   [rows, in_features]
//
// weight:
//   [in_features, out_features]
//
// bias:
//   [out_features]
//
// ============================================================

Tensor linear_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    bool use_bias,
    std::size_t in_features,
    std::size_t out_features
) {
    const std::size_t rows =
        input.numel() /
        in_features;

    std::vector<std::size_t>
        dimensions =
            input.shape().dimensions();

    dimensions.back() =
        out_features;

    Tensor result =
        Tensor::zeros(
            Shape(dimensions),
            DType::Float32,
            input.device()
        );

    const float* x =
        input.data_as<float>();

    const float* w =
        weight.data_as<float>();

    const float* b =
        use_bias
            ? bias.data_as<float>()
            : nullptr;

    float* y =
        result.data_as<float>();

    for (std::size_t row = 0;
         row < rows;
         ++row) {

        for (std::size_t out = 0;
             out < out_features;
             ++out) {

            double value =
                0.0;

            for (std::size_t in = 0;
                 in < in_features;
                 ++in) {

                value +=
                    static_cast<double>(
                        x[
                            row *
                            in_features +
                            in
                        ]
                    ) *
                    static_cast<double>(
                        w[
                            in *
                            out_features +
                            out
                        ]
                    );
            }

            if (b != nullptr) {

                value +=
                    static_cast<double>(
                        b[out]
                    );
            }

            y[
                row *
                out_features +
                out
            ] =
                static_cast<float>(
                    value
                );
        }
    }

    return result;
}

// ============================================================
// AUTOGRAD
// ============================================================

std::shared_ptr<AutogradNode>
make_feed_forward_node(
    const Tensor& input,
    const Tensor& input_weight,
    const Tensor& input_bias,
    const Tensor& output_weight,
    const Tensor& output_bias,
    std::size_t embed_dim,
    std::size_t hidden_dim,
    bool use_bias
) {
    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{
            input,
            input_weight,
            input_bias,
            output_weight,
            output_bias
        },

        [
            input,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
            embed_dim,
            hidden_dim,
            use_bias
        ](
            const Tensor& gradient
        ) {

            if (gradient.dtype() !=
                DType::Float32) {

                throw std::runtime_error(
                    "FeedForward backward: "
                    "gradient must be Float32"
                );
            }

            if (gradient.shape() !=
                input.shape()) {

                throw std::runtime_error(
                    "FeedForward backward: "
                    "gradient shape mismatch"
                );
            }

            const std::size_t rows =
                input.numel() /
                embed_dim;

            const float* x =
                input.data_as<float>();

            const float* w1 =
                input_weight.data_as<float>();

            const float* w2 =
                output_weight.data_as<float>();

            const float* b1 =
                use_bias
                    ? input_bias.data_as<float>()
                    : nullptr;

            const float* dy =
                gradient.data_as<float>();

            // ------------------------------------------------
            // Recreate first linear output.
            // ------------------------------------------------

            std::vector<float> hidden(
                rows *
                hidden_dim,
                0.0f
            );

            for (std::size_t row = 0;
                 row < rows;
                 ++row) {

                for (std::size_t h = 0;
                     h < hidden_dim;
                     ++h) {

                    double value =
                        0.0;

                    for (std::size_t in = 0;
                         in < embed_dim;
                         ++in) {

                        value +=
                            static_cast<double>(
                                x[
                                    row *
                                    embed_dim +
                                    in
                                ]
                            ) *
                            static_cast<double>(
                                w1[
                                    in *
                                    hidden_dim +
                                    h
                                ]
                            );
                    }

                    if (b1 != nullptr) {

                        value +=
                            static_cast<double>(
                                b1[h]
                            );
                    }

                    hidden[
                        row *
                        hidden_dim +
                        h
                    ] =
                        static_cast<float>(
                            value
                        );
                }
            }

            // ------------------------------------------------
            // GELU values.
            // ------------------------------------------------

            std::vector<float> activated(
                rows *
                hidden_dim,
                0.0f
            );

            for (std::size_t i = 0;
                 i < activated.size();
                 ++i) {

                activated[i] =
                    gelu_value(
                        hidden[i]
                    );
            }

            // ------------------------------------------------
            // Gradient buffers.
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

            Tensor dw1;

            if (input_weight.requires_grad()) {

                dw1 =
                    Tensor::zeros(
                        input_weight.shape(),
                        DType::Float32,
                        input_weight.device()
                    );
            }

            Tensor db1;

            if (use_bias &&
                input_bias.requires_grad()) {

                db1 =
                    Tensor::zeros(
                        input_bias.shape(),
                        DType::Float32,
                        input_bias.device()
                    );
            }

            Tensor dw2;

            if (output_weight.requires_grad()) {

                dw2 =
                    Tensor::zeros(
                        output_weight.shape(),
                        DType::Float32,
                        output_weight.device()
                    );
            }

            Tensor db2;

            if (use_bias &&
                output_bias.requires_grad()) {

                db2 =
                    Tensor::zeros(
                        output_bias.shape(),
                        DType::Float32,
                        output_bias.device()
                    );
            }

            float* dx_data =
                input.requires_grad()
                    ? dx.data_as<float>()
                    : nullptr;

            float* dw1_data =
                input_weight.requires_grad()
                    ? dw1.data_as<float>()
                    : nullptr;

            float* db1_data =
                use_bias &&
                input_bias.requires_grad()
                    ? db1.data_as<float>()
                    : nullptr;

            float* dw2_data =
                output_weight.requires_grad()
                    ? dw2.data_as<float>()
                    : nullptr;

            float* db2_data =
                use_bias &&
                output_bias.requires_grad()
                    ? db2.data_as<float>()
                    : nullptr;

            // ------------------------------------------------
            // Backward rows.
            // ------------------------------------------------

            std::vector<float> d_hidden(
                rows *
                hidden_dim,
                0.0f
            );

            // ------------------------------------------------
            // Second Linear backward.
            //
            // y = activated * W2 + b2
            // ------------------------------------------------

            for (std::size_t row = 0;
                 row < rows;
                 ++row) {

                for (std::size_t out = 0;
                     out < embed_dim;
                     ++out) {

                    const float upstream =
                        dy[
                            row *
                            embed_dim +
                            out
                        ];

                    if (db2_data != nullptr) {

                        db2_data[out] +=
                            upstream;
                    }

                    for (std::size_t h = 0;
                         h < hidden_dim;
                         ++h) {

                        if (dw2_data != nullptr) {

                            dw2_data[
                                h *
                                embed_dim +
                                out
                            ] +=
                                activated[
                                    row *
                                    hidden_dim +
                                    h
                                ] *
                                upstream;
                        }

                        d_hidden[
                            row *
                            hidden_dim +
                            h
                        ] +=
                            w2[
                                h *
                                embed_dim +
                                out
                            ] *
                            upstream;
                    }
                }
            }

            // ------------------------------------------------
            // GELU backward.
            // ------------------------------------------------

            std::vector<float> d_pre_hidden(
                rows *
                hidden_dim,
                0.0f
            );

            for (std::size_t i = 0;
                 i < d_pre_hidden.size();
                 ++i) {

                d_pre_hidden[i] =
                    d_hidden[i] *
                    gelu_derivative(
                        hidden[i]
                    );
            }

            // ------------------------------------------------
            // First Linear backward.
            //
            // hidden = x * W1 + b1
            // ------------------------------------------------

            for (std::size_t row = 0;
                 row < rows;
                 ++row) {

                for (std::size_t h = 0;
                     h < hidden_dim;
                     ++h) {

                    const float upstream =
                        d_pre_hidden[
                            row *
                            hidden_dim +
                            h
                        ];

                    if (db1_data != nullptr) {

                        db1_data[h] +=
                            upstream;
                    }

                    for (std::size_t in = 0;
                         in < embed_dim;
                         ++in) {

                        const float x_value =
                            x[
                                row *
                                embed_dim +
                                in
                            ];

                        if (dw1_data != nullptr) {

                            dw1_data[
                                in *
                                hidden_dim +
                                h
                            ] +=
                                x_value *
                                upstream;
                        }

                        if (dx_data != nullptr) {

                            dx_data[
                                row *
                                embed_dim +
                                in
                            ] +=
                                w1[
                                    in *
                                    hidden_dim +
                                    h
                                ] *
                                upstream;
                        }
                    }
                }
            }

            // ------------------------------------------------
            // Propagate input.
            // ------------------------------------------------

            if (input.requires_grad()) {

                input.accumulate_grad(
                    dx
                );

                if (input.grad_state()->grad_fn) {

                    input.grad_state()
                        ->grad_fn
                        ->backward(
                            dx
                        );
                }
            }

            // ------------------------------------------------
            // Propagate W1.
            // ------------------------------------------------

            if (input_weight.requires_grad()) {

                input_weight.accumulate_grad(
                    dw1
                );

                if (input_weight.grad_state()->grad_fn) {

                    input_weight.grad_state()
                        ->grad_fn
                        ->backward(
                            dw1
                        );
                }
            }

            // ------------------------------------------------
            // Propagate b1.
            // ------------------------------------------------

            if (use_bias &&
                input_bias.requires_grad()) {

                input_bias.accumulate_grad(
                    db1
                );

                if (input_bias.grad_state()->grad_fn) {

                    input_bias.grad_state()
                        ->grad_fn
                        ->backward(
                            db1
                        );
                }
            }

            // ------------------------------------------------
            // Propagate W2.
            // ------------------------------------------------

            if (output_weight.requires_grad()) {

                output_weight.accumulate_grad(
                    dw2
                );

                if (output_weight.grad_state()->grad_fn) {

                    output_weight.grad_state()
                        ->grad_fn
                        ->backward(
                            dw2
                        );
                }
            }

            // ------------------------------------------------
            // Propagate b2.
            // ------------------------------------------------

            if (use_bias &&
                output_bias.requires_grad()) {

                output_bias.accumulate_grad(
                    db2
                );

                if (output_bias.grad_state()->grad_fn) {

                    output_bias.grad_state()
                        ->grad_fn
                        ->backward(
                            db2
                        );
                }
            }
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

FeedForward::FeedForward(
    std::size_t embed_dim,
    std::size_t hidden_dim,
    bool use_bias
)
    : embed_dim_(embed_dim),
      hidden_dim_(hidden_dim),
      use_bias_(use_bias),
      input_weight_(),
      input_bias_(),
      output_weight_(),
      output_bias_() {

    if (embed_dim == 0) {

        throw std::invalid_argument(
            "FeedForward: "
            "embed_dim must be greater than zero"
        );
    }

    if (hidden_dim == 0) {

        throw std::invalid_argument(
            "FeedForward: "
            "hidden_dim must be greater than zero"
        );
    }

    input_weight_ =
        Tensor::empty(
            {
                embed_dim_,
                hidden_dim_
            },
            DType::Float32,
            Device::cpu()
        );

    output_weight_ =
        Tensor::empty(
            {
                hidden_dim_,
                embed_dim_
            },
            DType::Float32,
            Device::cpu()
        );

    if (use_bias_) {

        input_bias_ =
            Tensor::zeros(
                {hidden_dim_},
                DType::Float32,
                Device::cpu()
            );

        output_bias_ =
            Tensor::zeros(
                {embed_dim_},
                DType::Float32,
                Device::cpu()
            );
    }

    input_weight_.requires_grad_(true);
    output_weight_.requires_grad_(true);

    if (use_bias_) {

        input_bias_.requires_grad_(true);
        output_bias_.requires_grad_(true);
    }

    reset_parameters();
}

// ============================================================
// RESET PARAMETERS
// ============================================================

void FeedForward::reset_parameters() {

    const float input_limit =
        std::sqrt(
            6.0f /
            static_cast<float>(
                embed_dim_ +
                hidden_dim_
            )
        );

    const float output_limit =
        std::sqrt(
            6.0f /
            static_cast<float>(
                hidden_dim_ +
                embed_dim_
            )
        );

    std::mt19937 generator(
        0x46464E4Cu
    );

    std::uniform_real_distribution<float>
        input_distribution(
            -input_limit,
            input_limit
        );

    std::uniform_real_distribution<float>
        output_distribution(
            -output_limit,
            output_limit
        );

    float* w1 =
        input_weight_.data_as<float>();

    for (std::size_t i = 0;
         i < input_weight_.numel();
         ++i) {

        w1[i] =
            input_distribution(
                generator
            );
    }

    float* w2 =
        output_weight_.data_as<float>();

    for (std::size_t i = 0;
         i < output_weight_.numel();
         ++i) {

        w2[i] =
            output_distribution(
                generator
            );
    }

    if (use_bias_) {

        float* b1 =
            input_bias_.data_as<float>();

        float* b2 =
            output_bias_.data_as<float>();

        for (std::size_t i = 0;
             i < input_bias_.numel();
             ++i) {

            b1[i] =
                0.0f;
        }

        for (std::size_t i = 0;
             i < output_bias_.numel();
             ++i) {

            b2[i] =
                0.0f;
        }
    }
}

// ============================================================
// FORWARD
// ============================================================

Tensor FeedForward::forward(
    const Tensor& input
) const {

    validate_input(
        input
    );

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim_) {

        std::ostringstream message;

        message
            << "FeedForward::forward: "
            << "input last dimension must be "
            << embed_dim_
            << ", got "
            << features;

        throw std::runtime_error(
            message.str()
        );
    }

    if (input.shape()[
            input.ndim() - 2
        ] == 0) {

        throw std::runtime_error(
            "FeedForward::forward: "
            "sequence length must be greater than zero"
        );
    }

    Tensor hidden =
        linear_forward(
            input,
            input_weight_,
            input_bias_,
            use_bias_,
            embed_dim_,
            hidden_dim_
        );

    float* hidden_data =
        hidden.data_as<float>();

    for (std::size_t i = 0;
         i < hidden.numel();
         ++i) {

        hidden_data[i] =
            gelu_value(
                hidden_data[i]
            );
    }

    Tensor result =
        linear_forward(
            hidden,
            output_weight_,
            output_bias_,
            use_bias_,
            hidden_dim_,
            embed_dim_
        );

    if (input.requires_grad() ||
        input_weight_.requires_grad() ||
        output_weight_.requires_grad() ||
        (
            use_bias_ &&
            (
                input_bias_.requires_grad() ||
                output_bias_.requires_grad()
            )
        )) {

        result.set_grad_fn(
            make_feed_forward_node(
                input,
                input_weight_,
                input_bias_,
                output_weight_,
                output_bias_,
                embed_dim_,
                hidden_dim_,
                use_bias_
            )
        );
    }

    return result;
}

// ============================================================
// PARAMETERS
// ============================================================

const Tensor& FeedForward::input_weight() const {
    return input_weight_;
}

Tensor& FeedForward::input_weight() {
    return input_weight_;
}

const Tensor& FeedForward::input_bias() const {
    return input_bias_;
}

Tensor& FeedForward::input_bias() {
    return input_bias_;
}

const Tensor& FeedForward::output_weight() const {
    return output_weight_;
}

Tensor& FeedForward::output_weight() {
    return output_weight_;
}

const Tensor& FeedForward::output_bias() const {
    return output_bias_;
}

Tensor& FeedForward::output_bias() {
    return output_bias_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t FeedForward::embed_dim() const {
    return embed_dim_;
}

std::size_t FeedForward::hidden_dim() const {
    return hidden_dim_;
}

bool FeedForward::has_bias() const {
    return use_bias_;
}

} // namespace venla
