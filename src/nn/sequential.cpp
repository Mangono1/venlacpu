#include "venla/nn/sequential.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace venla {

// ============================================================
// LAYER
// ============================================================

Sequential::Layer::Layer(
    LayerType layer_type
)
    : type(layer_type),
      linear(nullptr) {
}

Sequential::Layer::Layer(
    std::unique_ptr<Linear> linear_layer
)
    : type(LayerType::Linear),
      linear(std::move(linear_layer)) {
}

// ============================================================
// CONSTRUCTOR
// ============================================================

Sequential::Sequential()
    : layers_() {
}

// ============================================================
// ADD LINEAR
// ============================================================

Sequential& Sequential::add_linear(
    std::size_t in_features,
    std::size_t out_features,
    bool use_bias
) {

    auto layer =
        std::make_unique<Linear>(
            in_features,
            out_features,
            use_bias
        );

    layers_.emplace_back(
        std::move(layer)
    );

    return *this;
}

// ============================================================
// ADD RELU
// ============================================================

Sequential& Sequential::add_relu() {

    layers_.emplace_back(
        LayerType::ReLU
    );

    return *this;
}

// ============================================================
// ADD SIGMOID
// ============================================================

Sequential& Sequential::add_sigmoid() {

    layers_.emplace_back(
        LayerType::Sigmoid
    );

    return *this;
}

// ============================================================
// ADD TANH
// ============================================================

Sequential& Sequential::add_tanh() {

    layers_.emplace_back(
        LayerType::Tanh
    );

    return *this;
}

// ============================================================
// FORWARD
//
// The output of one layer becomes the input of the next.
//
// Example:
//
// input
//   |
// Linear
//   |
// ReLU
//   |
// Linear
//   |
// ReLU
//   |
// output
//
// ============================================================

Tensor Sequential::forward(
    const Tensor& input
) const {

    Tensor current = input;

    for (const Layer& layer : layers_) {

        switch (layer.type) {

            case LayerType::Linear:

                if (!layer.linear) {
                    throw std::runtime_error(
                        "Sequential::forward: "
                        "Linear layer is not initialized"
                    );
                }

                current =
                    layer.linear->forward(
                        current
                    );

                break;

            case LayerType::ReLU:

                current =
                    relu(
                        current
                    );

                break;

            case LayerType::Sigmoid:

                current =
                    sigmoid(
                        current
                    );

                break;

            case LayerType::Tanh:

                current =
                    tanh(
                        current
                    );

                break;

            default:

                throw std::runtime_error(
                    "Sequential::forward: "
                    "unknown layer type"
                );
        }
    }

    return current;
}

// ============================================================
// SIZE
// ============================================================

std::size_t Sequential::size() const {
    return layers_.size();
}

// ============================================================
// EMPTY
// ============================================================

bool Sequential::empty() const {
    return layers_.empty();
}

// ============================================================
// LAYER TYPE
// ============================================================

Sequential::LayerType Sequential::layer_type(
    std::size_t index
) const {

    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Sequential::layer_type: "
            "layer index out of range"
        );
    }

    return layers_[index].type;
}

// ============================================================
// LINEAR CONST ACCESS
// ============================================================

const Linear& Sequential::linear(
    std::size_t index
) const {

    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Sequential::linear: "
            "layer index out of range"
        );
    }

    const Layer& layer =
        layers_[index];

    if (layer.type != LayerType::Linear ||
        !layer.linear) {

        throw std::runtime_error(
            "Sequential::linear: "
            "selected layer is not Linear"
        );
    }

    return *layer.linear;
}

// ============================================================
// LINEAR MUTABLE ACCESS
// ============================================================

Linear& Sequential::linear(
    std::size_t index
) {

    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Sequential::linear: "
            "layer index out of range"
        );
    }

    Layer& layer =
        layers_[index];

    if (layer.type != LayerType::Linear ||
        !layer.linear) {

        throw std::runtime_error(
            "Sequential::linear: "
            "selected layer is not Linear"
        );
    }

    return *layer.linear;
}

// ============================================================
// CLEAR
// ============================================================

void Sequential::clear() {
    layers_.clear();
}

} // namespace venla

