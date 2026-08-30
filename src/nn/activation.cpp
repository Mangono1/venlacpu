#include "venla/nn/activation.hpp"
#include "venla/autograd/ops.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void validate_activation_input(
    const Tensor& input,
    const char* operation
) {
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error(
            std::string(operation) +
            ": currently only Float32 is supported"
        );
    }

    if (!input.device().is_cpu()) {
        throw std::runtime_error(
            std::string(operation) +
            ": only CPU device is currently supported"
        );
    }
}

// ============================================================
// CREATE OUTPUT
// ============================================================

Tensor create_output(
    const Tensor& input
) {
    return Tensor::zeros(
        input.shape(),
        DType::Float32,
        input.device()
    );
}

// ============================================================
// PROPAGATE ACTIVATION GRADIENT
// ============================================================

void propagate_activation_gradient(
    const Tensor& input,
    const Tensor& gradient
) {
    if (!input.requires_grad()) {
        return;
    }

    input.accumulate_grad(
        gradient
    );

    if (input.grad_state()->grad_fn) {
        input.grad_state()
            ->grad_fn
            ->backward(
                gradient
            );
    }
}

} // namespace

// ============================================================
// RELU
//
// y = max(0, x)
//
// backward:
//
// dx = dy  jika x > 0
//      0   jika x <= 0
// ============================================================

Tensor relu(
    const Tensor& input
) {
    validate_activation_input(
        input,
        "relu"
    );

    Tensor output =
        create_output(input);

    const float* x =
        input.data_as<float>();

    float* y =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        y[i] =
            x[i] > 0.0f
                ? x[i]
                : 0.0f;
    }

    if (input.requires_grad()) {


        std::shared_ptr<AutogradNode> node =
            std::make_shared<AutogradNode>(
                std::vector<Tensor>{input},

                [input](
                    const Tensor& gradient
                ) mutable {

                    if (gradient.shape() !=
                        input.shape()) {

                        throw std::runtime_error(
                            "relu backward: "
                            "gradient shape mismatch"
                        );
                    }

                    Tensor grad_input =
                        Tensor::zeros(
                            input.shape(),
                            DType::Float32,
                            input.device()
                        );

                    const float* x =
                        input.data_as<float>();

                    const float* upstream =
                        gradient.data_as<float>();

                    float* gx =
                        grad_input.data_as<float>();

                    for (std::size_t i = 0;
                         i < input.numel();
                         ++i) {

                        gx[i] =
                            x[i] > 0.0f
                                ? upstream[i]
                                : 0.0f;
                    }

                    propagate_activation_gradient(
                        input,
                        grad_input
                    );
                }
            );

        output.set_grad_fn(node);
    }

    return output;
}

// ============================================================
// SIGMOID
//
// y = 1 / (1 + exp(-x))
//
// backward:
//
// dx = dy * y * (1-y)
// ============================================================

Tensor sigmoid(
    const Tensor& input
) {
    validate_activation_input(
        input,
        "sigmoid"
    );

    Tensor output =
        create_output(input);

    const float* x =
        input.data_as<float>();

    float* y =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        const float value =
            x[i];

        if (value >= 0.0f) {

            const float z =
                std::exp(-value);

            y[i] =
                1.0f /
                (1.0f + z);

        } else {

            const float z =
                std::exp(value);

            y[i] =
                z /
                (1.0f + z);
        }
    }

    if (input.requires_grad()) {


        std::shared_ptr<AutogradNode> node =
            std::make_shared<AutogradNode>(
                std::vector<Tensor>{input},

                [input, output](
                    const Tensor& gradient
                ) mutable {

                    if (gradient.shape() !=
                        input.shape()) {

                        throw std::runtime_error(
                            "sigmoid backward: "
                            "gradient shape mismatch"
                        );
                    }

                    Tensor grad_input =
                        Tensor::zeros(
                            input.shape(),
                            DType::Float32,
                            input.device()
                        );

                    const float* y =
                        output.data_as<float>();

                    const float* upstream =
                        gradient.data_as<float>();

                    float* gx =
                        grad_input.data_as<float>();

                    for (std::size_t i = 0;
                         i < input.numel();
                         ++i) {

                        gx[i] =
                            upstream[i] *
                            y[i] *
                            (1.0f - y[i]);
                    }

                    propagate_activation_gradient(
                        input,
                        grad_input
                    );
                }
            );

        output.set_grad_fn(node);
    }

    return output;
}

// ============================================================
// TANH
//
// y = tanh(x)
//
// backward:
//
// dx = dy * (1-y²)
// ============================================================

Tensor tanh(
    const Tensor& input
) {
    validate_activation_input(
        input,
        "tanh"
    );

    Tensor output =
        create_output(input);

    const float* x =
        input.data_as<float>();

    float* y =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        y[i] =
            std::tanh(x[i]);
    }

    if (input.requires_grad()) {


        std::shared_ptr<AutogradNode> node =
            std::make_shared<AutogradNode>(
                std::vector<Tensor>{input},

                [input, output](
                    const Tensor& gradient
                ) mutable {

                    if (gradient.shape() !=
                        input.shape()) {

                        throw std::runtime_error(
                            "tanh backward: "
                            "gradient shape mismatch"
                        );
                    }

                    Tensor grad_input =
                        Tensor::zeros(
                            input.shape(),
                            DType::Float32,
                            input.device()
                        );

                    const float* y =
                        output.data_as<float>();

                    const float* upstream =
                        gradient.data_as<float>();

                    float* gx =
                        grad_input.data_as<float>();

                    for (std::size_t i = 0;
                         i < input.numel();
                         ++i) {

                        gx[i] =
                            upstream[i] *
                            (1.0f -
                             y[i] * y[i]);
                    }

                    propagate_activation_gradient(
                        input,
                        grad_input
                    );
                }
            );

        output.set_grad_fn(node);
    }

    return output;
}

} // namespace venla
