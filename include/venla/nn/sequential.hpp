#pragma once

#include "venla/nn/activation.hpp"
#include "venla/nn/linear.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace venla {

// ============================================================
// SEQUENTIAL
//
// Executes neural-network layers in order.
//
// Example:
//
//   Sequential model;
//
//   model.add_linear(3, 8);
//   model.add_relu();
//   model.add_linear(8, 4);
//   model.add_relu();
//
//   Tensor output = model.forward(input);
//
// Supported layers:
//   - Linear
//   - ReLU
//   - Sigmoid
//   - Tanh
//
// The implementation is intentionally CPU-first and
// dependency-free.
//
// ============================================================

class Sequential {
public:

    // --------------------------------------------------------
    // Layer types
    // --------------------------------------------------------

    enum class LayerType {
        Linear,
        ReLU,
        Sigmoid,
        Tanh
    };

    // --------------------------------------------------------
    // Construction
    // --------------------------------------------------------

    Sequential();

    // --------------------------------------------------------
    // Layer creation
    // --------------------------------------------------------

    Sequential& add_linear(
        std::size_t in_features,
        std::size_t out_features,
        bool use_bias = true
    );

    Sequential& add_relu();

    Sequential& add_sigmoid();

    Sequential& add_tanh();

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t size() const;

    bool empty() const;

    LayerType layer_type(
        std::size_t index
    ) const;

    // --------------------------------------------------------
    // Linear access
    //
    // Throws std::out_of_range if the selected layer is not
    // a Linear layer or index is invalid.
    // --------------------------------------------------------

    const Linear& linear(
        std::size_t index
    ) const;

    Linear& linear(
        std::size_t index
    );

    // --------------------------------------------------------
    // Remove all layers
    // --------------------------------------------------------

    void clear();

private:

    struct Layer {
        LayerType type;

        std::unique_ptr<Linear> linear;

        explicit Layer(
            LayerType layer_type
        );

        explicit Layer(
            std::unique_ptr<Linear> linear_layer
        );
    };

    std::vector<Layer> layers_;
};

} // namespace venla

