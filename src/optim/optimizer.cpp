#include "venla/optim/optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace venla {

// ============================================================
// VALIDATION
// ============================================================

namespace {

void validate_parameter(
    const Tensor& parameter
) {
    if (parameter.empty()) {
        throw std::invalid_argument(
            "Optimizer: parameter tensor is empty"
        );
    }

    if (parameter.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Optimizer: only Float32 parameters are supported"
        );
    }

    if (!parameter.device().is_cpu()) {
        throw std::runtime_error(
            "Optimizer: only CPU parameters are currently supported"
        );
    }

    if (!parameter.requires_grad()) {
        throw std::runtime_error(
            "Optimizer: parameter must require gradients"
        );
    }
}

} // namespace

// ============================================================
// OPTIMIZER
// ============================================================

void Optimizer::add_parameter(
    Tensor& parameter
) {
    validate_parameter(parameter);

    for (Tensor* existing : parameters_) {
        if (existing == &parameter) {
            return;
        }
    }

    parameters_.push_back(
        &parameter
    );
}

void Optimizer::add_parameters(
    const std::vector<Tensor*>& parameters
) {
    for (Tensor* parameter : parameters) {

        if (parameter == nullptr) {
            throw std::invalid_argument(
                "Optimizer::add_parameters: "
                "null parameter"
            );
        }

        add_parameter(*parameter);
    }
}

std::size_t Optimizer::parameter_count() const {
    return parameters_.size();
}

void Optimizer::zero_grad() {

    for (Tensor* parameter : parameters_) {

        if (parameter == nullptr) {
            continue;
        }

        parameter->zero_grad();
    }
}

const std::vector<Tensor*>&
Optimizer::parameters() const {
    return parameters_;
}

std::vector<Tensor*>&
Optimizer::parameters() {
    return parameters_;
}

// ============================================================
// SGD CONSTRUCTOR
// ============================================================

SGD::SGD(
    float learning_rate,
    float momentum,
    float weight_decay
)
    : learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      momentum_buffers_() {

    if (!std::isfinite(learning_rate) ||
        learning_rate <= 0.0f) {

        throw std::invalid_argument(
            "SGD: learning_rate must be finite and greater than zero"
        );
    }

    if (!std::isfinite(momentum) ||
        momentum < 0.0f ||
        momentum >= 1.0f) {

        throw std::invalid_argument(
            "SGD: momentum must be in [0, 1)"
        );
    }

    if (!std::isfinite(weight_decay) ||
        weight_decay < 0.0f) {

        throw std::invalid_argument(
            "SGD: weight_decay must be finite and non-negative"
        );
    }
}

// ============================================================
// SGD STEP
// ============================================================

void SGD::step() {

    for (Tensor* parameter : parameters()) {

        if (parameter == nullptr) {
            throw std::runtime_error(
                "SGD::step: null parameter"
            );
        }

        validate_parameter(*parameter);

        if (!parameter->has_grad()) {
            continue;
        }

        const std::size_t count =
            parameter->numel();

        float* data =
            parameter->data_as<float>();

        const float* gradient =
            parameter->grad().data_as<float>();

        std::vector<float>* momentum_buffer =
            nullptr;

        if (momentum_ > 0.0f) {

            std::vector<float>& buffer =
                momentum_buffers_[parameter];

            if (buffer.size() != count) {
                buffer.assign(
                    count,
                    0.0f
                );
            }

            momentum_buffer =
                &buffer;
        }

        for (std::size_t i = 0;
             i < count;
             ++i) {

            float grad =
                gradient[i];

            if (weight_decay_ != 0.0f) {
                grad +=
                    weight_decay_ *
                    data[i];
            }

            if (momentum_buffer != nullptr) {

                (*momentum_buffer)[i] =
                    momentum_ *
                    (*momentum_buffer)[i]
                    +
                    grad;

                grad =
                    (*momentum_buffer)[i];
            }

            data[i] -=
                learning_rate_ *
                grad;
        }
    }
}

// ============================================================
// SGD METADATA
// ============================================================

float SGD::learning_rate() const {
    return learning_rate_;
}

float SGD::momentum() const {
    return momentum_;
}

float SGD::weight_decay() const {
    return weight_decay_;
}

void SGD::set_learning_rate(
    float value
) {
    if (!std::isfinite(value) ||
        value <= 0.0f) {

        throw std::invalid_argument(
            "SGD::set_learning_rate: "
            "value must be finite and greater than zero"
        );
    }

    learning_rate_ =
        value;
}

// ============================================================
// ADAM CONSTRUCTOR
// ============================================================

Adam::Adam(
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay
)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      step_count_(0),
      states_() {

    if (!std::isfinite(learning_rate) ||
        learning_rate <= 0.0f) {

        throw std::invalid_argument(
            "Adam: learning_rate must be finite and greater than zero"
        );
    }

    if (!std::isfinite(beta1) ||
        beta1 < 0.0f ||
        beta1 >= 1.0f) {

        throw std::invalid_argument(
            "Adam: beta1 must be in [0, 1)"
        );
    }

    if (!std::isfinite(beta2) ||
        beta2 < 0.0f ||
        beta2 >= 1.0f) {

        throw std::invalid_argument(
            "Adam: beta2 must be in [0, 1)"
        );
    }

    if (!std::isfinite(epsilon) ||
        epsilon <= 0.0f) {

        throw std::invalid_argument(
            "Adam: epsilon must be finite and greater than zero"
        );
    }

    if (!std::isfinite(weight_decay) ||
        weight_decay < 0.0f) {

        throw std::invalid_argument(
            "Adam: weight_decay must be finite and non-negative"
        );
    }
}

// ============================================================
// ADAM STEP
// ============================================================

void Adam::step() {

    ++step_count_;

    const float bias_correction_1 =
        1.0f -
        std::pow(
            beta1_,
            static_cast<float>(step_count_)
        );

    const float bias_correction_2 =
        1.0f -
        std::pow(
            beta2_,
            static_cast<float>(step_count_)
        );

    for (Tensor* parameter : parameters()) {

        if (parameter == nullptr) {
            throw std::runtime_error(
                "Adam::step: null parameter"
            );
        }

        validate_parameter(*parameter);

        if (!parameter->has_grad()) {
            continue;
        }

        const std::size_t count =
            parameter->numel();

        float* data =
            parameter->data_as<float>();

        const float* gradient =
            parameter->grad().data_as<float>();

        State& state =
            states_[parameter];

        if (state.first_moment.size() != count) {

            state.first_moment.assign(
                count,
                0.0f
            );

            state.second_moment.assign(
                count,
                0.0f
            );
        }

        for (std::size_t i = 0;
             i < count;
             ++i) {

            float grad =
                gradient[i];

            if (weight_decay_ != 0.0f) {
                grad +=
                    weight_decay_ *
                    data[i];
            }

            state.first_moment[i] =
                beta1_ *
                state.first_moment[i]
                +
                (1.0f - beta1_) *
                grad;

            state.second_moment[i] =
                beta2_ *
                state.second_moment[i]
                +
                (1.0f - beta2_) *
                grad *
                grad;

            const float first_corrected =
                state.first_moment[i] /
                bias_correction_1;

            const float second_corrected =
                state.second_moment[i] /
                bias_correction_2;

            data[i] -=
                learning_rate_ *
                first_corrected /
                (
                    std::sqrt(
                        second_corrected
                    ) +
                    epsilon_
                );
        }
    }
}

// ============================================================
// ADAM METADATA
// ============================================================

float Adam::learning_rate() const {
    return learning_rate_;
}

float Adam::beta1() const {
    return beta1_;
}

float Adam::beta2() const {
    return beta2_;
}

float Adam::epsilon() const {
    return epsilon_;
}

float Adam::weight_decay() const {
    return weight_decay_;
}

std::size_t Adam::step_count() const {
    return step_count_;
}

void Adam::set_learning_rate(
    float value
) {
    if (!std::isfinite(value) ||
        value <= 0.0f) {

        throw std::invalid_argument(
            "Adam::set_learning_rate: "
            "value must be finite and greater than zero"
        );
    }

    learning_rate_ =
        value;
}

} // namespace venla
