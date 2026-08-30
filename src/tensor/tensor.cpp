#include "venla/tensor/tensor.hpp"

#include "venla/autograd/autograd.hpp"
#include "venla/math/operations.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace venla {

// ============================================================
// CONSTRUCTORS
// ============================================================

Tensor::Tensor()
    : shape_(),
      stride_(),
      dtype_(DType::Float32),
      device_(Device::cpu()),
      storage_(std::make_shared<Storage>()),
      grad_state_(std::make_shared<GradState>()) {
}

Tensor::Tensor(
    const Shape& shape,
    DType dtype,
    const Device& device
)
    : shape_(shape),
      stride_(Stride::contiguous(shape)),
      dtype_(dtype),
      device_(device),
      storage_(std::make_shared<Storage>(
          shape.numel() * dtype_size(dtype)
      )),
      grad_state_(std::make_shared<GradState>()) {
}

// ============================================================
// FACTORIES
// ============================================================

Tensor Tensor::empty(
    const Shape& shape,
    DType dtype,
    const Device& device
) {
    return Tensor(
        shape,
        dtype,
        device
    );
}

Tensor Tensor::zeros(
    const Shape& shape,
    DType dtype,
    const Device& device
) {
    Tensor result(
        shape,
        dtype,
        device
    );

    if (result.nbytes() != 0) {
        std::memset(
            result.data(),
            0,
            result.nbytes()
        );
    }

    return result;
}

Tensor Tensor::ones(
    const Shape& shape,
    DType dtype,
    const Device& device
) {
    Tensor result(
        shape,
        dtype,
        device
    );

    if (dtype != DType::Float32) {
        throw std::runtime_error(
            "Tensor::ones currently initializes Float32 only"
        );
    }

    float* values =
        result.data_as<float>();

    for (std::size_t i = 0;
         i < result.numel();
         ++i) {

        values[i] = 1.0f;
    }

    return result;
}

// ============================================================
// AUTOGRAD
// ============================================================

void Tensor::requires_grad_(
    bool enabled
) {
    grad_state_->requires_grad =
        enabled;

    if (!enabled) {
        grad_state_->grad_fn.reset();

        grad_state_->grad.reset();

        grad_state_->is_leaf = true;
    }
}

bool Tensor::requires_grad() const {
    return grad_state_->requires_grad;
}

bool Tensor::is_leaf() const {
    return grad_state_->is_leaf;
}

bool Tensor::has_grad() const {
    return
        grad_state_->grad != nullptr;
}

const Tensor& Tensor::grad() const {
    if (!grad_state_->grad) {
        throw std::runtime_error(
            "Tensor::grad: gradient is not available"
        );
    }

    return *grad_state_->grad;
}

Tensor& Tensor::grad() {
    if (!grad_state_->grad) {
        throw std::runtime_error(
            "Tensor::grad: gradient is not available"
        );
    }

    return *grad_state_->grad;
}

void Tensor::zero_grad() {

    if (!grad_state_->grad) {
        return;
    }

    Tensor& gradient =
        *grad_state_->grad;

    if (gradient.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Tensor::zero_grad: only Float32 supported"
        );
    }

    float* values =
        gradient.data_as<float>();

    for (std::size_t i = 0;
         i < gradient.numel();
         ++i) {

        values[i] = 0.0f;
    }
}

std::shared_ptr<GradState>
Tensor::grad_state() const {
    return grad_state_;
}

void Tensor::set_grad_fn(
    const std::shared_ptr<AutogradNode>& node
) {
    grad_state_->grad_fn =
        node;

    grad_state_->is_leaf =
        false;

    grad_state_->requires_grad =
        true;
}

// ============================================================
// ACCUMULATE GRADIENT
// ============================================================

void Tensor::accumulate_grad(
    const Tensor& gradient
) const {
    if (!requires_grad()) {
        return;
    }

    if (gradient.dtype() != DType::Float32) {
        throw std::runtime_error(
            "Tensor::accumulate_grad: only Float32 supported"
        );
    }

    if (shape() != gradient.shape()) {
        throw std::runtime_error(
            "Tensor::accumulate_grad: gradient shape mismatch"
        );
    }

    if (!grad_state_->grad) {

        grad_state_->grad =
            std::make_shared<Tensor>(
                Tensor::zeros(
                    shape(),
                    DType::Float32,
                    device()
                )
            );
    }

    Tensor& destination =
        *grad_state_->grad;

    const float* source =
        gradient.data_as<float>();

    float* target =
        destination.data_as<float>();

    for (std::size_t i = 0;
         i < numel();
         ++i) {

        target[i] += source[i];
    }
}

// ============================================================
// BACKWARD
// ============================================================

void Tensor::backward() {

    if (numel() != 1) {
        throw std::runtime_error(
            "Tensor::backward: implicit gradient requires scalar tensor"
        );
    }

    Tensor gradient =
        Tensor::ones(
            shape(),
            DType::Float32,
            device()
        );

    backward(gradient);
}

void Tensor::backward(
    const Tensor& gradient
) {
    if (!requires_grad()) {
        throw std::runtime_error(
            "Tensor::backward: tensor does not require gradients"
        );
    }

    if (gradient.shape() != shape()) {
        throw std::runtime_error(
            "Tensor::backward: gradient shape mismatch"
        );
    }

    accumulate_grad(gradient);

    if (grad_state_->grad_fn) {
        grad_state_->grad_fn->backward(
            gradient
        );
    }
}

// ============================================================
// METADATA
// ============================================================

const Shape& Tensor::shape() const {
    return shape_;
}

const Stride& Tensor::stride() const {
    return stride_;
}

DType Tensor::dtype() const {
    return dtype_;
}

const Device& Tensor::device() const {
    return device_;
}

std::size_t Tensor::ndim() const {
    return shape_.ndim();
}

std::size_t Tensor::rank() const {
    return shape_.rank();
}

std::size_t Tensor::numel() const {
    return shape_.numel();
}

std::size_t Tensor::nbytes() const {
    return storage_->bytes();
}

bool Tensor::is_contiguous() const {
    return stride_.is_contiguous(
        shape_
    );
}

bool Tensor::empty() const {
    return storage_->empty();
}

void* Tensor::data() {
    return storage_->data();
}

const void* Tensor::data() const {
    return storage_->data();
}

std::string Tensor::info() const {

    std::ostringstream out;

    out
        << "Tensor("
        << "shape=" << shape_.to_string()
        << ", stride=" << stride_.to_string()
        << ", dtype=" << dtype_name(dtype_)
        << ", device=" << device_.to_string()
        << ", numel=" << numel()
        << ", nbytes=" << nbytes()
        << ", contiguous="
        << (
            is_contiguous()
                ? "true"
                : "false"
        )
        << ", requires_grad="
        << (
            requires_grad()
                ? "true"
                : "false"
        )
        << ", has_grad="
        << (
            has_grad()
                ? "true"
                : "false"
        )
        << ")";

    return out.str();
}

} // namespace venla
