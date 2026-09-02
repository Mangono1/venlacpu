#include "venla/training/checkpoint.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float a,
    float b,
    float tolerance,
    const std::string& message
) {
    if (std::fabs(a - b) > tolerance) {
        throw std::runtime_error(
            message +
            " | a=" + std::to_string(a) +
            " b=" + std::to_string(b)
        );
    }
}

std::vector<float> copy_tensor(const venla::Tensor& tensor) {
    require(
        tensor.dtype() == venla::DType::Float32,
        "Checkpoint test requires Float32"
    );

    const float* data = tensor.data_as<float>();

    require(
        data != nullptr,
        "Tensor data pointer is null"
    );

    return std::vector<float>(
        data,
        data + tensor.numel()
    );
}

void set_gradient(
    venla::Tensor& tensor,
    float value
) {
    require(
        tensor.dtype() == venla::DType::Float32,
        "Gradient requires Float32 tensor"
    );

    if (!tensor.requires_grad()) {
        tensor.requires_grad_(true);
    }

    venla::Tensor gradient = venla::Tensor::zeros(
        tensor.shape(),
        tensor.dtype(),
        tensor.device()
    );

    float* gradient_data =
        gradient.data_as<float>();

    require(
        gradient_data != nullptr,
        "Gradient data pointer is null"
    );

    for (std::size_t i = 0;
         i < gradient.numel();
         ++i) {
        gradient_data[i] = value;
    }

    tensor.backward(gradient);

    require(
        tensor.has_grad(),
        "Gradient was not created"
    );
}

bool tensors_equal(
    const venla::Tensor& a,
    const venla::Tensor& b,
    float tolerance = 1e-6f
) {
    if (a.dtype() != b.dtype()) {
        return false;
    }

    if (a.shape() != b.shape()) {
        return false;
    }

    if (a.numel() != b.numel()) {
        return false;
    }

    const float* ad = a.data_as<float>();
    const float* bd = b.data_as<float>();

    if (ad == nullptr || bd == nullptr) {
        return false;
    }

    for (std::size_t i = 0;
         i < a.numel();
         ++i) {

        if (std::fabs(ad[i] - bd[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

} // namespace

int main() {
    const std::string checkpoint_path =
        "venlacpu_checkpoint_test.bin";

    try {
        std::cout
            << "============================================================\n"
            << " VENLACPU 2.3.1 — CHECKPOINT ENGINE TEST\n"
            << "============================================================\n";

        /*
         * ========================================================
         * MODEL A
         * ========================================================
         *
         * VENLACPU LanguageModel memakai constructor langsung:
         *
         *   vocab_size
         *   max_seq_len
         *   embed_dim
         *   num_heads
         *   hidden_dim
         *   num_layers
         *   use_bias
         */
        venla::LanguageModel model_a(
            32,
            8,
            8,
            2,
            16,
            1,
            true
        );

        std::vector<venla::Tensor*> params_a =
            model_a.parameters();

        require(
            !params_a.empty(),
            "Model A has no parameters"
        );

        std::cout
            << "[OK] Model A parameters = "
            << params_a.size()
            << "\n";

        /*
         * ========================================================
         * OPTIMIZER A
         * ========================================================
         */
        venla::Adam optimizer_a(
            0.001f,
            0.9f,
            0.999f,
            1e-8f,
            0.0f
        );

        optimizer_a.add_parameters(params_a);

        require(
            optimizer_a.parameter_count() ==
            params_a.size(),
            "Optimizer A parameter count mismatch"
        );

        /*
         * ========================================================
         * CREATE GRADIENT
         * ========================================================
         */
        for (venla::Tensor* parameter : params_a) {
            require(
                parameter != nullptr,
                "Null parameter in Model A"
            );

            set_gradient(
                *parameter,
                0.01f
            );
        }

        /*
         * Adam step #1 creates optimizer state.
         */
        optimizer_a.step();

        require(
            optimizer_a.step_count() == 1,
            "Adam state was not initialized"
        );

        std::cout
            << "[OK] Adam state initialized\n";

        /*
         * ========================================================
         * SAVE CHECKPOINT
         * ========================================================
         */
        venla::CheckpointMetadata metadata;

        metadata.epoch = 7;
        metadata.global_step = 123;
        metadata.best_eval_loss = 1.2345f;
        metadata.bad_epochs = 2;

        venla::CheckpointEngine::save(
            checkpoint_path,
            model_a,
            optimizer_a,
            metadata
        );

        std::cout
            << "[OK] checkpoint saved\n";

        /*
         * ========================================================
         * MODEL B
         * ========================================================
         */
        venla::LanguageModel model_b(
            32,
            8,
            8,
            2,
            16,
            1,
            true
        );

        std::vector<venla::Tensor*> params_b =
            model_b.parameters();

        require(
            params_b.size() == params_a.size(),
            "Fresh Model B parameter count mismatch"
        );

        /*
         * ========================================================
         * OPTIMIZER B
         * ========================================================
         */
        venla::Adam optimizer_b(
            0.001f,
            0.9f,
            0.999f,
            1e-8f,
            0.0f
        );

        optimizer_b.add_parameters(params_b);

        /*
         * ========================================================
         * LOAD
         * ========================================================
         */
        venla::CheckpointMetadata loaded =
            venla::CheckpointEngine::load(
                checkpoint_path,
                model_b,
                optimizer_b
            );

        std::cout
            << "[OK] checkpoint loaded\n";

        /*
         * ========================================================
         * METADATA
         * ========================================================
         */
        require(
            loaded.epoch == metadata.epoch,
            "Epoch mismatch"
        );

        require(
            loaded.global_step == metadata.global_step,
            "Global step mismatch"
        );

        require_close(
            loaded.best_eval_loss,
            metadata.best_eval_loss,
            1e-6f,
            "Best eval loss mismatch"
        );

        require(
            loaded.bad_epochs == metadata.bad_epochs,
            "Bad epochs mismatch"
        );

        std::cout
            << "[OK] metadata restored\n";

        /*
         * ========================================================
         * MODEL PARAMETERS
         * ========================================================
         *
         * parameters() -> Tensor*
         * jadi harus didereference.
         */
        for (std::size_t i = 0;
             i < params_a.size();
             ++i) {

            require(
                params_a[i] != nullptr,
                "Null Model A parameter"
            );

            require(
                params_b[i] != nullptr,
                "Null Model B parameter"
            );

            require(
                tensors_equal(
                    *params_a[i],
                    *params_b[i]
                ),
                "Parameter mismatch at index " +
                std::to_string(i)
            );
        }

        std::cout
            << "[OK] model parameters restored\n";

        /*
         * ========================================================
         * OPTIMIZER STATE
         * ========================================================
         */
        require(
            optimizer_b.step_count() ==
            optimizer_a.step_count(),
            "Adam step_count mismatch"
        );

        std::cout
            << "[OK] optimizer state restored\n"
            << "     Adam step = "
            << optimizer_b.step_count()
            << "\n";

        /*
         * ========================================================
         * RESUMED TRAINING TEST
         * ========================================================
         *
         * Kedua model menerima gradient IDENTIK.
         *
         * Jika state Adam berhasil dipulihkan,
         * step berikutnya harus menghasilkan parameter
         * yang sama.
         */
        for (venla::Tensor* parameter : params_a) {
            require(
                parameter != nullptr,
                "Null parameter A"
            );

            parameter->zero_grad();

            set_gradient(
                *parameter,
                0.02f
            );
        }

        for (venla::Tensor* parameter : params_b) {
            require(
                parameter != nullptr,
                "Null parameter B"
            );

            parameter->zero_grad();

            set_gradient(
                *parameter,
                0.02f
            );
        }

        optimizer_a.step();
        optimizer_b.step();

        require(
            optimizer_a.step_count() == 2,
            "Optimizer A did not reach step 2"
        );

        require(
            optimizer_b.step_count() == 2,
            "Optimizer B did not reach step 2"
        );

        for (std::size_t i = 0;
             i < params_a.size();
             ++i) {

            require(
                tensors_equal(
                    *params_a[i],
                    *params_b[i],
                    1e-5f
                ),
                "Resumed optimizer mismatch at parameter " +
                std::to_string(i)
            );
        }

        std::cout
            << "[OK] resumed optimizer behavior matches\n";

        /*
         * ========================================================
         * CLEANUP
         * ========================================================
         */
        std::remove(
            checkpoint_path.c_str()
        );

        std::cout
            << "[OK] temporary checkpoint removed\n";

        std::cout
            << "\n============================================================\n"
            << " CHECKPOINT TEST PASSED\n"
            << "============================================================\n";

        return 0;

    } catch (const std::exception& e) {

        std::remove(
            checkpoint_path.c_str()
        );

        std::cerr
            << "\n============================================================\n"
            << " CHECKPOINT TEST FAILED\n"
            << "============================================================\n"
            << e.what()
            << "\n";

        return 1;
    }
}
