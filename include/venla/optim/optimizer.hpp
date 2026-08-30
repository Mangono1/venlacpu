#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace venla {

// ============================================================
// OPTIMIZER
//
// Base class untuk optimizer parameter.
//
// Parameter didaftarkan menggunakan:
//
//     optimizer.add_parameter(parameter);
//
// Optimizer bekerja langsung pada storage Tensor sehingga
// perubahan parameter dilakukan tanpa membuat graph autograd.
//
// ============================================================

class Optimizer {
public:

    virtual ~Optimizer() = default;

    // --------------------------------------------------------
    // PARAMETER REGISTRATION
    // --------------------------------------------------------

    void add_parameter(Tensor& parameter);

    void add_parameters(
        const std::vector<Tensor*>& parameters
    );

    std::size_t parameter_count() const;

    // --------------------------------------------------------
    // GRADIENT
    // --------------------------------------------------------

    void zero_grad();

    // --------------------------------------------------------
    // UPDATE
    // --------------------------------------------------------

    virtual void step() = 0;

protected:

    const std::vector<Tensor*>& parameters() const;

    std::vector<Tensor*>& parameters();

private:

    std::vector<Tensor*> parameters_;
};

// ============================================================
// SGD
//
// Standard Stochastic Gradient Descent:
//
//     p = p - lr * grad
//
// Dengan momentum:
//
//     v = momentum * v + grad
//     p = p - lr * v
//
// Weight decay:
//
//     grad = grad + weight_decay * p
//
// ============================================================

class SGD : public Optimizer {
public:

    explicit SGD(
        float learning_rate = 0.001f,
        float momentum = 0.0f,
        float weight_decay = 0.0f
    );

    void step() override;

    float learning_rate() const;

    float momentum() const;

    float weight_decay() const;

    void set_learning_rate(float value);

private:

    float learning_rate_;

    float momentum_;

    float weight_decay_;

    std::unordered_map<
        const Tensor*,
        std::vector<float>
    > momentum_buffers_;
};

// ============================================================
// ADAM
//
// Adam:
//
//     m = beta1 * m + (1-beta1) * g
//     v = beta2 * v + (1-beta2) * g²
//
//     m_hat = m / (1-beta1^t)
//     v_hat = v / (1-beta2^t)
//
//     p = p - lr * m_hat / (sqrt(v_hat) + eps)
//
// Weight decay menggunakan bentuk L2:
//
//     g = g + weight_decay * p
//
// ============================================================

class Adam : public Optimizer {
public:

    explicit Adam(
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weight_decay = 0.0f
    );

    void step() override;

    float learning_rate() const;

    float beta1() const;

    float beta2() const;

    float epsilon() const;

    float weight_decay() const;

    std::size_t step_count() const;

    void set_learning_rate(float value);

private:

    struct State {
        std::vector<float> first_moment;
        std::vector<float> second_moment;
    };

    float learning_rate_;

    float beta1_;

    float beta2_;

    float epsilon_;

    float weight_decay_;

    std::size_t step_count_;

    std::unordered_map<
        const Tensor*,
        State
    > states_;
};

} // namespace venla
