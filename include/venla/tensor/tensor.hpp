#pragma once

#include "venla/core/device.hpp"
#include "venla/core/dtype.hpp"
#include "venla/core/shape.hpp"
#include "venla/core/storage.hpp"
#include "venla/core/stride.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace venla {

struct AutogradNode;

struct GradState {
    bool requires_grad = false;

    std::shared_ptr<class Tensor> grad;

    std::shared_ptr<AutogradNode> grad_fn;

    bool is_leaf = true;
};

class Tensor {
public:

    Tensor();

    Tensor(
        const Shape& shape,
        DType dtype = DType::Float32,
        const Device& device = Device::cpu()
    );

    static Tensor zeros(
        const Shape& shape,
        DType dtype = DType::Float32,
        const Device& device = Device::cpu()
    );

    static Tensor ones(
        const Shape& shape,
        DType dtype = DType::Float32,
        const Device& device = Device::cpu()
    );

    static Tensor empty(
        const Shape& shape,
        DType dtype = DType::Float32,
        const Device& device = Device::cpu()
    );

    // ========================================================
    // AUTOGRAD
    // ========================================================

    void requires_grad_(bool enabled = true);

    bool requires_grad() const;

    bool is_leaf() const;

    bool has_grad() const;

    const Tensor& grad() const;

    Tensor& grad();

    void zero_grad();

    void backward();

    void backward(
        const Tensor& gradient
    );

    // ========================================================
    // INTERNAL AUTOGRAD SUPPORT
    // ========================================================

    std::shared_ptr<GradState> grad_state() const;

    void set_grad_fn(
        const std::shared_ptr<AutogradNode>& node
    );

    void accumulate_grad(
        const Tensor& gradient
    ) const;

    // ========================================================
    // TENSOR METADATA
    // ========================================================

    const Shape& shape() const;

    const Stride& stride() const;

    DType dtype() const;

    const Device& device() const;

    std::size_t ndim() const;

    std::size_t rank() const;

    std::size_t numel() const;

    std::size_t nbytes() const;

    bool is_contiguous() const;

    bool empty() const;

    void* data();

    const void* data() const;

    template <typename T>
    T* data_as() {
        return static_cast<T*>(data());
    }

    template <typename T>
    const T* data_as() const {
        return static_cast<const T*>(data());
    }

    std::string info() const;

private:

    Shape shape_;

    Stride stride_;

    DType dtype_;

    Device device_;

    std::shared_ptr<Storage> storage_;

    std::shared_ptr<GradState> grad_state_;
};

} // namespace venla
