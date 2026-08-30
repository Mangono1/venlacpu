#include "venla/nn/mse_loss.hpp"
#include "venla/autograd/ops.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void validate_tensor(
    const Tensor& tensor,
    const char* name
) {
    if (tensor.dtype() != DType::Float32) {

        throw std::runtime_error(
            std::string(name) +
            ": currently only Float32 is supported"
        );
    }

    if (!tensor.device().is_cpu()) {

        throw std::runtime_error(
            std::string(name) +
            ": only CPU device is currently supported"
        );
    }
}

// ============================================================
// SHAPE VALIDATION
// ============================================================

void validate_shapes(
    const Tensor& prediction,
    const Tensor& target
) {
    if (prediction.shape() != target.shape()) {

        std::ostringstream message;

        message
            << "MSELoss: prediction and target "
            << "must have identical shapes. "
            << "prediction="
            << prediction.shape().to_string()
            << ", target="
            << target.shape().to_string();

        throw std::runtime_error(
            message.str()
        );
    }
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

MSELoss::MSELoss(
    Reduction reduction
)
    : reduction_(reduction) {
}

// ============================================================
// FORWARD
// ============================================================

Tensor MSELoss::forward(
    const Tensor& prediction,
    const Tensor& target
) const {

    validate_tensor(
        prediction,
        "MSELoss prediction"
    );

    validate_tensor(
        target,
        "MSELoss target"
    );

    validate_shapes(
        prediction,
        target
    );

    const float* prediction_data =
        prediction.data_as<float>();

    const float* target_data =
        target.data_as<float>();

    const std::size_t count =
        prediction.numel();

    // ========================================================
    // REDUCTION NONE
    // ========================================================

    if (reduction_ == Reduction::None) {

        Tensor result =
            Tensor::zeros(
                prediction.shape(),
                DType::Float32,
                prediction.device()
            );

        float* output =
            result.data_as<float>();

        for (std::size_t i = 0;
             i < count;
             ++i) {

            const float difference =
                prediction_data[i] -
                target_data[i];

            output[i] =
                difference *
                difference;
        }

        // ----------------------------------------------------
        // AUTOGRAD
        // ----------------------------------------------------

        if (prediction.requires_grad()) {

            result.set_grad_fn(
                std::make_shared<AutogradNode>(
                    std::vector<Tensor>{
                        prediction
                    },

                    [prediction, target](
                        const Tensor& gradient
                    ) {

                        if (gradient.shape() !=
                            prediction.shape()) {

                            throw std::runtime_error(
                                "MSELoss backward: "
                                "gradient shape mismatch"
                            );
                        }

                        Tensor grad_input =
                            Tensor::zeros(
                                prediction.shape(),
                                DType::Float32,
                                prediction.device()
                            );

                        const float* pv =
                            prediction.data_as<float>();

                        const float* tv =
                            target.data_as<float>();

                        const float* gv =
                            gradient.data_as<float>();

                        float* gi =
                            grad_input.data_as<float>();

                        for (std::size_t i = 0;
                             i < prediction.numel();
                             ++i) {

                            gi[i] =
                                2.0f *
                                (pv[i] - tv[i]) *
                                gv[i];
                        }

                        backward_tensor(
                            prediction,
                            grad_input
                        );
                    }
                )
            );
        }

        return result;
    }

    // ========================================================
    // SUM / MEAN
    // ========================================================

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            prediction.device()
        );

    float* output =
        result.data_as<float>();

    float total =
        0.0f;

    for (std::size_t i = 0;
         i < count;
         ++i) {

        const float difference =
            prediction_data[i] -
            target_data[i];

        total +=
            difference *
            difference;
    }

    // ========================================================
    // SUM
    // ========================================================

    if (reduction_ == Reduction::Sum) {

        output[0] =
            total;

        if (prediction.requires_grad()) {

            result.set_grad_fn(
                std::make_shared<AutogradNode>(
                    std::vector<Tensor>{
                        prediction
                    },

                    [prediction, target](
                        const Tensor& gradient
                    ) {

                        if (gradient.numel() != 1) {

                            throw std::runtime_error(
                                "MSELoss backward: "
                                "SUM gradient must be scalar"
                            );
                        }

                        const float upstream =
                            gradient.data_as<float>()[0];

                        Tensor grad_input =
                            Tensor::zeros(
                                prediction.shape(),
                                DType::Float32,
                                prediction.device()
                            );

                        const float* pv =
                            prediction.data_as<float>();

                        const float* tv =
                            target.data_as<float>();

                        float* gi =
                            grad_input.data_as<float>();

                        for (std::size_t i = 0;
                             i < prediction.numel();
                             ++i) {

                            gi[i] =
                                2.0f *
                                (pv[i] - tv[i]) *
                                upstream;
                        }

                        backward_tensor(
                            prediction,
                            grad_input
                        );
                    }
                )
            );
        }

        return result;
    }

    // ========================================================
    // MEAN
    // ========================================================

    if (reduction_ == Reduction::Mean) {

        if (count == 0) {

            output[0] =
                0.0f;
        }
        else {

            output[0] =
                total /
                static_cast<float>(count);
        }

        if (prediction.requires_grad()) {

            result.set_grad_fn(
                std::make_shared<AutogradNode>(
                    std::vector<Tensor>{
                        prediction
                    },

                    [prediction, target, count](
                        const Tensor& gradient
                    ) {

                        if (gradient.numel() != 1) {

                            throw std::runtime_error(
                                "MSELoss backward: "
                                "MEAN gradient must be scalar"
                            );
                        }

                        const float upstream =
                            gradient.data_as<float>()[0];

                        Tensor grad_input =
                            Tensor::zeros(
                                prediction.shape(),
                                DType::Float32,
                                prediction.device()
                            );

                        const float* pv =
                            prediction.data_as<float>();

                        const float* tv =
                            target.data_as<float>();

                        float* gi =
                            grad_input.data_as<float>();

                        if (count == 0) {

                            for (std::size_t i = 0;
                                 i < prediction.numel();
                                 ++i) {

                                gi[i] =
                                    0.0f;
                            }
                        }
                        else {

                            const float scale =
                                upstream /
                                static_cast<float>(count);

                            for (std::size_t i = 0;
                                 i < prediction.numel();
                                 ++i) {

                                gi[i] =
                                    2.0f *
                                    (pv[i] - tv[i]) *
                                    scale;
                            }
                        }

                        backward_tensor(
                            prediction,
                            grad_input
                        );
                    }
                )
            );
        }

        return result;
    }

    throw std::runtime_error(
        "MSELoss: unsupported reduction"
    );
}

// ============================================================
// METADATA
// ============================================================

Reduction MSELoss::reduction() const {
    return reduction_;
}

} // namespace venla
