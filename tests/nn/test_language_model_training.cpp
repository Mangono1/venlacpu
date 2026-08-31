#include "venla/nn/cross_entropy_loss.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/tensor/tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace venla;

namespace {

Tensor make_tokens(
    const std::vector<std::int32_t>& values
) {
    Tensor tensor =
        Tensor::empty(
            {values.size()},
            DType::Int32,
            Device::cpu()
        );

    std::int32_t* data =
        tensor.data_as<std::int32_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {

        data[i] = values[i];
    }

    return tensor;
}

Tensor make_targets(
    const std::vector<std::int32_t>& values
) {
    Tensor tensor =
        Tensor::empty(
            {values.size()},
            DType::Int32,
            Device::cpu()
        );

    std::int32_t* data =
        tensor.data_as<std::int32_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {

        data[i] = values[i];
    }

    return tensor;
}

float parameter_checksum(
    LanguageModel& model
) {
    float checksum = 0.0f;

    std::vector<Tensor*> parameters =
        model.parameters();

    for (Tensor* parameter : parameters) {

        assert(parameter != nullptr);

        const float* data =
            parameter->data_as<float>();

        for (std::size_t i = 0;
             i < parameter->numel();
             ++i) {

            checksum +=
                std::fabs(data[i]);
        }
    }

    return checksum;
}

bool parameters_have_gradients(
    LanguageModel& model
) {
    std::vector<Tensor*> parameters =
        model.parameters();

    bool found_gradient =
        false;

    for (Tensor* parameter : parameters) {

        assert(parameter != nullptr);

        assert(
            parameter->requires_grad()
        );

        if (parameter->has_grad()) {

            const Tensor& gradient =
                parameter->grad();

            assert(
                gradient.shape() ==
                parameter->shape()
            );

            assert(
                gradient.dtype() ==
                DType::Float32
            );

            const float* data =
                gradient.data_as<float>();

            for (std::size_t i = 0;
                 i < gradient.numel();
                 ++i) {

                assert(
                    std::isfinite(data[i])
                );
            }

            found_gradient = true;
        }
    }

    return found_gradient;
}

void test_parameter_registration() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    assert(
        !parameters.empty()
    );

    for (Tensor* parameter : parameters) {

        assert(
            parameter != nullptr
        );

        assert(
            parameter->requires_grad()
        );

        assert(
            parameter->dtype() ==
            DType::Float32
        );

        assert(
            parameter->device().is_cpu()
        );
    }

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        parameters
    );

    assert(
        optimizer.parameter_count() ==
        parameters.size()
    );
}

void test_forward_loss_backward() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Tensor input =
        make_tokens(
            {
                1,
                2,
                3,
                4
            }
        );

    Tensor targets =
        make_targets(
            {
                2,
                3,
                4,
                5
            }
        );

    Tensor logits =
        model.forward(
            input
        );

    assert(
        logits.dtype() ==
        DType::Float32
    );

    assert(
        logits.shape().ndim() ==
        2
    );

    assert(
        logits.shape()[0] ==
        4
    );

    assert(
        logits.shape()[1] ==
        16
    );

    CrossEntropyLoss loss;

    Tensor loss_value =
        loss.forward(
            logits,
            targets
        );

    assert(
        loss_value.ndim() == 0
    );

    assert(
        loss_value.requires_grad()
    );

    assert(
        loss_value.grad_state() != nullptr
    );

    assert(
        loss_value.grad_state()->grad_fn != nullptr
    );

    assert(
        std::isfinite(
            loss_value.data_as<float>()[0]
        )
    );

    loss_value.backward();

    assert(
        parameters_have_gradients(
            model
        )
    );
}

void test_adam_updates_parameters() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        parameters
    );

    Tensor input =
        make_tokens(
            {
                1,
                2,
                3,
                4
            }
        );

    Tensor targets =
        make_targets(
            {
                2,
                3,
                4,
                5
            }
        );

    const float before =
        parameter_checksum(
            model
        );

    Tensor logits =
        model.forward(
            input
        );

    CrossEntropyLoss loss;

    Tensor loss_value =
        loss.forward(
            logits,
            targets
        );

    assert(
        std::isfinite(
            loss_value.data_as<float>()[0]
        )
    );

    loss_value.backward();

    assert(
        parameters_have_gradients(
            model
        )
    );

    optimizer.step();

    const float after =
        parameter_checksum(
            model
        );

    assert(
        std::isfinite(before)
    );

    assert(
        std::isfinite(after)
    );

    assert(
        before != after
    );

    assert(
        optimizer.step_count() == 1
    );
}

void test_multiple_training_steps() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    Tensor input =
        make_tokens(
            {
                1,
                2,
                3,
                4
            }
        );

    Tensor targets =
        make_targets(
            {
                2,
                3,
                4,
                5
            }
        );

    float first_loss =
        0.0f;

    float last_loss =
        0.0f;

    for (std::size_t step = 0;
         step < 3;
         ++step) {

        optimizer.zero_grad();

        Tensor logits =
            model.forward(
                input
            );

        CrossEntropyLoss loss;

        Tensor loss_value =
            loss.forward(
                logits,
                targets
            );

        assert(
            std::isfinite(
                loss_value.data_as<float>()[0]
            )
        );

        if (step == 0) {

            first_loss =
                loss_value.data_as<float>()[0];
        }

        loss_value.backward();

        assert(
            parameters_have_gradients(
                model
            )
        );

        optimizer.step();

        last_loss =
            loss_value.data_as<float>()[0];

        assert(
            optimizer.step_count() ==
            step + 1
        );
    }

    assert(
        std::isfinite(first_loss)
    );

    assert(
        std::isfinite(last_loss)
    );
}

void test_zero_grad_after_training() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    Tensor input =
        make_tokens(
            {
                1,
                2,
                3,
                4
            }
        );

    Tensor targets =
        make_targets(
            {
                2,
                3,
                4,
                5
            }
        );

    Tensor logits =
        model.forward(
            input
        );

    CrossEntropyLoss loss;

    Tensor loss_value =
        loss.forward(
            logits,
            targets
        );

    loss_value.backward();

    assert(
        parameters_have_gradients(
            model
        )
    );

    optimizer.zero_grad();

    std::vector<Tensor*> parameters =
        model.parameters();

    for (Tensor* parameter : parameters) {

        assert(
            parameter != nullptr
        );

        assert(
            !parameter->has_grad()
        );
    }
}

} // namespace

int main() {

    test_parameter_registration();

    test_forward_loss_backward();

    test_adam_updates_parameters();

    test_multiple_training_steps();

    test_zero_grad_after_training();

    std::cout
        << "VENLACPU LanguageModel training tests passed."
        << std::endl;

    return 0;
}
