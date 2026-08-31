#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace venla {

// ============================================================
// CROSS ENTROPY LOSS
//
// Causal Language Model:
//
//     logits:
//         [seq, vocab]
//         [batch, seq, vocab]
//
//     targets:
//         [seq]
//         [batch, seq]
//
// Loss:
//
//     L = -log(softmax(logits)[target])
//
// Stable implementation:
//
//     L = -x_target
//         + max(x)
//         + log(sum(exp(x - max(x))))
//
// Gradient:
//
//     dL/dx = softmax(x) - one_hot(target)
//
// Untuk reduction mean:
//
//     dL/dx /= jumlah token valid
//
// ignore_index digunakan untuk mengabaikan token tertentu,
// biasanya token <PAD>.
//
// CPU / Float32 logits.
// Targets harus Int32 atau Int64.
// ============================================================

class CrossEntropyLoss {
public:

    explicit CrossEntropyLoss(
        std::int64_t ignore_index = -100,
        bool reduction_mean = true
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& logits,
        const Tensor& targets
    ) const;

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::int64_t ignore_index() const;

    bool reduction_mean() const;

private:

    std::int64_t ignore_index_;

    bool reduction_mean_;
};

} // namespace venla
