#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// MSE LOSS
//
// Mean Squared Error:
//
//   loss = (prediction - target)^2
//
// Reduction:
//   None -> element-wise loss
//   Sum  -> sum of all losses
//   Mean -> average of all losses
//
// Supported:
//   Float32
//   CPU
//
// prediction and target must have identical shapes.
//
// ============================================================

enum class Reduction {
    None,
    Sum,
    Mean
};

class MSELoss {
public:

    explicit MSELoss(
        Reduction reduction = Reduction::Mean
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& prediction,
        const Tensor& target
    ) const;

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    Reduction reduction() const;

private:

    Reduction reduction_;
};

} // namespace venla
