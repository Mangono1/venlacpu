#include "venla/nn/linear.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace venla {

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
            "Linear: in_features must be greater than zero"
        );
    }

    if (out_features == 0) {
        throw std::invalid_argument(
            "Linear: out_features must be greater than zero"
        );
    }

    weight_ =
        Tensor::empty(
            {in_features_, out_features_},
            DType::Float32,
            Device::cpu()
        );

    if (use_bias_) {
        bias_ =
            Tensor::empty(
                {out_features_},
                DType::Float32,
                Device::cpu()
            );
    }
    else {
        bias_ = Tensor();
    }

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
//   limit = sqrt(6 / (fan_in + fan_out))
//
//   weight ~ U(-limit, +limit)
//
// Bias:
//
//   0
//
// A fixed seed is intentionally used for now so that:
//
//   Linear(3, 4)
//   Linear(3, 4)
//
// starts with reproducible parameters.
//
// Later, VENLACPU can expose configurable RNG generators/seeds.
// ============================================================

void Linear::reset_parameters() {

    if (weight_.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Linear::reset_parameters: weight must be Float32"
        );
    }

    if (!weight_.device().is_cpu()) {
        throw std::runtime_error(
            "Linear::reset_parameters: only CPU device is currently supported"
        );
    }

    const float fan_in =
        static_cast<float>(in_features_);

    const float fan_out =
        static_cast<float>(out_features_);

    const float limit =
        std::sqrt(
            6.0f /
            (fan_in + fan_out)
        );

    // Deterministic engine for reproducible initialization.
    std::mt19937 generator(0x56454E4Cu);

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
            distribution(generator);
    }

    if (use_bias_) {

        if (bias_.dtype() != DType::Float32) {
            throw std::runtime_error(
                "Linear::reset_parameters: bias must be Float32"
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
//   [..., in_features]
//
// Weight:
//
//   [in_features, out_features]
//
// Output:
//
//   [..., out_features]
//
// The implementation treats every leading dimension as a
// batch dimension.
//
// Examples:
//
//   [3]          -> [4]
//   [2,3]        -> [2,4]
//   [5,2,3]      -> [5,2,4]
//   [2,4,2,3]    -> [2,4,2,4]
//
// ============================================================

Tensor Linear::forward(
    const Tensor& input
) const {

    if (input.ndim() == 0) {
        throw std::runtime_error(
            "Linear::forward: input must have at least 1 dimension"
        );
    }

    if (input.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Linear::forward: currently only Float32 is supported"
        );
    }

    if (!input.device().is_cpu()) {
        throw std::runtime_error(
            "Linear::forward: only CPU device is currently supported"
        );
    }

    const std::size_t input_features =
        input.shape()[input.ndim() - 1];

    if (input_features != in_features_) {

        std::ostringstream message;

        message
            << "Linear::forward: input last dimension must be "
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
    // 1D
    //
    // [in_features]
    //
    // -> [out_features]
    // --------------------------------------------------------

    if (input.ndim() == 1) {

        Tensor result =
            Tensor::zeros(
                {out_features_},
                DType::Float32,
                input.device()
            );

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

        for (std::size_t out = 0;
             out < out_features_;
             ++out) {

            float value = 0.0f;

            for (std::size_t in = 0;
                 in < in_features_;
                 ++in) {

                value +=
                    x[in] *
                    w[
                        in * out_features_ +
                        out
                    ];
            }

            if (b != nullptr) {
                value += b[out];
            }

            y[out] =
                value;
        }

        return result;
    }

    // --------------------------------------------------------
    // ND
    //
    // Flatten all leading dimensions into rows.
    //
    // [2,3]
    //       -> 2 rows
    //
    // [5,2,3]
    //       -> 10 rows
    //
    // [2,4,2,3]
    //       -> 16 rows
    //
    // Every row contains exactly in_features values.
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

    const std::size_t rows =
        input.numel() /
        in_features_;

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

            float value = 0.0f;

            for (std::size_t in = 0;
                 in < in_features_;
                 ++in) {

                value +=
                    x_row[in] *
                    w[
                        in * out_features_ +
                        out
                    ];
            }

            if (b != nullptr) {
                value += b[out];
            }

            y_row[out] =
                value;
        }
    }

    return result;
}

// ============================================================
// WEIGHT
// ============================================================

const Tensor& Linear::weight() const {
    return weight_;
}

Tensor& Linear::weight() {
    return weight_;
}

// ============================================================
// BIAS
// ============================================================

const Tensor& Linear::bias() const {
    return bias_;
}

Tensor& Linear::bias() {
    return bias_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t Linear::in_features() const {
    return in_features_;
}

std::size_t Linear::out_features() const {
    return out_features_;
}

bool Linear::has_bias() const {
    return use_bias_;
}

} // namespace venla
