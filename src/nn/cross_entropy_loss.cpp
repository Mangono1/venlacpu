#include "venla/nn/cross_entropy_loss.hpp"

#include "venla/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace venla {

namespace {

// ============================================================
// TARGET ACCESS
// ============================================================

std::int64_t read_target(
    const Tensor& targets,
    std::size_t index
) {
    if (targets.dtype() == DType::Int32) {

        return static_cast<std::int64_t>(
            targets.data_as<std::int32_t>()[index]
        );
    }

    if (targets.dtype() == DType::Int64) {

        return targets.data_as<std::int64_t>()[index];
    }

    throw std::runtime_error(
        "CrossEntropyLoss: "
        "targets must be Int32 or Int64"
    );
}

// ============================================================
// PROPAGATE GRADIENT TO PARENT
//
// Local version of the autograd propagation mechanism.
//
// Gradient shape is identical to parent shape, so no broadcast
// reduction is required here.
// ============================================================

void propagate_to_parent(
    const Tensor& parent,
    const Tensor& gradient
) {
    if (!parent.requires_grad()) {
        return;
    }

    if (gradient.shape() != parent.shape()) {

        throw std::runtime_error(
            "CrossEntropyLoss backward: "
            "gradient shape mismatch"
        );
    }

    parent.accumulate_grad(
        gradient
    );

    if (parent.grad_state() &&
        parent.grad_state()->grad_fn) {

        parent.grad_state()
            ->grad_fn
            ->backward(
                gradient
            );
    }
}

// ============================================================
// VALIDATE INPUT
// ============================================================

void validate_inputs(
    const Tensor& logits,
    const Tensor& targets
) {
    // --------------------------------------------------------
    // Logits rank
    // --------------------------------------------------------

    if (logits.ndim() != 2 &&
        logits.ndim() != 3) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "logits must be 2D [seq, vocab] or "
            "3D [batch, seq, vocab]"
        );
    }

    // --------------------------------------------------------
    // Logits dtype
    // --------------------------------------------------------

    if (logits.dtype() != DType::Float32) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "logits must be Float32"
        );
    }

    // --------------------------------------------------------
    // CPU only
    // --------------------------------------------------------

    if (!logits.device().is_cpu()) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "only CPU device is currently supported"
        );
    }

    if (!targets.device().is_cpu()) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "targets must be on CPU"
        );
    }

    // --------------------------------------------------------
    // Targets dtype
    // --------------------------------------------------------

    if (targets.dtype() != DType::Int32 &&
        targets.dtype() != DType::Int64) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "targets must be Int32 or Int64"
        );
    }

    // --------------------------------------------------------
    // Shape relationship
    // --------------------------------------------------------

    if (logits.ndim() == 2) {

        // logits:
        //
        // [seq, vocab]
        //
        // targets:
        //
        // [seq]

        if (targets.ndim() != 1) {

            throw std::runtime_error(
                "CrossEntropyLoss: "
                "2D logits require 1D targets"
            );
        }

        if (targets.shape()[0] !=
            logits.shape()[0]) {

            throw std::runtime_error(
                "CrossEntropyLoss: "
                "target sequence length does not "
                "match logits sequence length"
            );
        }
    }

    else {

        // logits:
        //
        // [batch, seq, vocab]
        //
        // targets:
        //
        // [batch, seq]

        if (targets.ndim() != 2) {

            throw std::runtime_error(
                "CrossEntropyLoss: "
                "3D logits require 2D targets"
            );
        }

        if (targets.shape()[0] !=
            logits.shape()[0]) {

            throw std::runtime_error(
                "CrossEntropyLoss: "
                "target batch dimension does not "
                "match logits"
            );
        }

        if (targets.shape()[1] !=
            logits.shape()[1]) {

            throw std::runtime_error(
                "CrossEntropyLoss: "
                "target sequence dimension does not "
                "match logits"
            );
        }
    }

    // --------------------------------------------------------
    // Empty tensors
    // --------------------------------------------------------

    if (targets.numel() == 0) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "targets cannot be empty"
        );
    }

    if (logits.shape()[logits.ndim() - 1] == 0) {

        throw std::runtime_error(
            "CrossEntropyLoss: "
            "vocabulary dimension must be greater than zero"
        );
    }
}

// ============================================================
// GET LOGIT ROW OFFSET
//
// 2D:
//
//     [seq, vocab]
//
// 3D:
//
//     [batch, seq, vocab]
//
// Every target corresponds to one contiguous vocab row.
// ============================================================

std::size_t row_offset(
    const Tensor& logits,
    std::size_t token_index
) {
    const std::size_t vocab_size =
        logits.shape()[logits.ndim() - 1];

    return token_index * vocab_size;
}

// ============================================================
// CROSS ENTROPY AUTOGRAD NODE
// ============================================================

std::shared_ptr<AutogradNode>
make_cross_entropy_node(
    const Tensor& logits,
    const Tensor& targets,
    std::int64_t ignore_index,
    bool reduction_mean
) {
    if (!logits.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{logits},

        [
            logits,
            targets,
            ignore_index,
            reduction_mean
        ](const Tensor& gradient) mutable {

            // ------------------------------------------------
            // Loss output is scalar.
            // ------------------------------------------------

            if (gradient.numel() != 1) {

                throw std::runtime_error(
                    "CrossEntropyLoss backward: "
                    "expected scalar upstream gradient"
                );
            }

            const std::size_t vocab_size =
                logits.shape()[
                    logits.ndim() - 1
                ];

            const std::size_t token_count =
                targets.numel();

            const float upstream =
                gradient.data_as<float>()[0];

            const float* logits_data =
                logits.data_as<float>();

            Tensor grad_logits =
                Tensor::zeros(
                    logits.shape(),
                    DType::Float32,
                    logits.device()
                );

            float* gradient_data =
                grad_logits.data_as<float>();

            // ------------------------------------------------
            // Count valid targets.
            // ------------------------------------------------

            std::size_t valid_count = 0;

            for (std::size_t token = 0;
                 token < token_count;
                 ++token) {

                const std::int64_t target =
                    read_target(
                        targets,
                        token
                    );

                if (target == ignore_index) {
                    continue;
                }

                if (target < 0 ||
                    target >=
                        static_cast<std::int64_t>(
                            vocab_size
                        )) {

                    std::ostringstream message;

                    message
                        << "CrossEntropyLoss backward: "
                        << "target index "
                        << target
                        << " out of range [0, "
                        << vocab_size
                        << ")";

                    throw std::out_of_range(
                        message.str()
                    );
                }

                ++valid_count;
            }

            // ------------------------------------------------
            // Mean denominator.
            //
            // If every target is ignored, all gradients remain
            // zero.
            // ------------------------------------------------

            float scale = upstream;

            if (reduction_mean) {

                if (valid_count == 0) {

                    propagate_to_parent(
                        logits,
                        grad_logits
                    );

                    return;
                }

                scale /=
                    static_cast<float>(
                        valid_count
                    );
            }

            // ------------------------------------------------
            // Calculate gradient row by row.
            //
            // gradient =
            //
            //     softmax(logits)
            //
            // then:
            //
            //     gradient[target] -= 1
            //
            // ------------------------------------------------

            for (std::size_t token = 0;
                 token < token_count;
                 ++token) {

                const std::int64_t target =
                    read_target(
                        targets,
                        token
                    );

                if (target == ignore_index) {
                    continue;
                }

                const std::size_t offset =
                    row_offset(
                        logits,
                        token
                    );

                // ------------------------------------------------
                // Maximum for numerical stability.
                // ------------------------------------------------

                float maximum =
                    logits_data[offset];

                for (std::size_t j = 1;
                     j < vocab_size;
                     ++j) {

                    if (logits_data[
                            offset + j
                        ] > maximum) {

                        maximum =
                            logits_data[
                                offset + j
                            ];
                    }
                }

                // ------------------------------------------------
                // exp(x - max)
                // ------------------------------------------------

                float exponential_sum =
                    0.0f;

                for (std::size_t j = 0;
                     j < vocab_size;
                     ++j) {

                    const float value =
                        std::exp(
                            logits_data[
                                offset + j
                            ] - maximum
                        );

                    gradient_data[
                        offset + j
                    ] =
                        value;

                    exponential_sum +=
                        value;
                }

                // ------------------------------------------------
                // softmax
                // ------------------------------------------------

                const float inverse_sum =
                    1.0f /
                    exponential_sum;

                for (std::size_t j = 0;
                     j < vocab_size;
                     ++j) {

                    gradient_data[
                        offset + j
                    ] *=
                        inverse_sum *
                        scale;
                }

                // ------------------------------------------------
                // Target correction.
                // ------------------------------------------------

                gradient_data[
                    offset +
                    static_cast<std::size_t>(
                        target
                    )
                ] -= scale;
            }

            propagate_to_parent(
                logits,
                grad_logits
            );
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

CrossEntropyLoss::CrossEntropyLoss(
    std::int64_t ignore_index,
    bool reduction_mean
)
    : ignore_index_(ignore_index),
      reduction_mean_(reduction_mean) {
}

// ============================================================
// FORWARD
// ============================================================

Tensor CrossEntropyLoss::forward(
    const Tensor& logits,
    const Tensor& targets
) const {

    validate_inputs(
        logits,
        targets
    );

    const std::size_t vocab_size =
        logits.shape()[
            logits.ndim() - 1
        ];

    const std::size_t token_count =
        targets.numel();

    const float* logits_data =
        logits.data_as<float>();

    // --------------------------------------------------------
    // First pass:
    //
    // validate targets and calculate loss.
    // --------------------------------------------------------

    double total_loss = 0.0;

    std::size_t valid_count = 0;

    for (std::size_t token = 0;
         token < token_count;
         ++token) {

        const std::int64_t target =
            read_target(
                targets,
                token
            );

        if (target == ignore_index_) {
            continue;
        }

        if (target < 0 ||
            target >=
                static_cast<std::int64_t>(
                    vocab_size
                )) {

            std::ostringstream message;

            message
                << "CrossEntropyLoss: "
                << "target index "
                << target
                << " out of range [0, "
                << vocab_size
                << ")";

            throw std::out_of_range(
                message.str()
            );
        }

        ++valid_count;

        const std::size_t offset =
            row_offset(
                logits,
                token
            );

        // ----------------------------------------------------
        // Stable log-sum-exp.
        // ----------------------------------------------------

        float maximum =
            logits_data[offset];

        for (std::size_t j = 1;
             j < vocab_size;
             ++j) {

            if (logits_data[
                    offset + j
                ] > maximum) {

                maximum =
                    logits_data[
                        offset + j
                    ];
            }
        }

        double exponential_sum =
            0.0;

        for (std::size_t j = 0;
             j < vocab_size;
             ++j) {

            exponential_sum +=
                std::exp(
                    static_cast<double>(
                        logits_data[
                            offset + j
                        ] - maximum
                    )
                );
        }

        const double log_sum_exp =
            static_cast<double>(
                maximum
            ) +
            std::log(
                exponential_sum
            );

        const double target_logit =
            static_cast<double>(
                logits_data[
                    offset +
                    static_cast<std::size_t>(
                        target
                    )
                ]
            );

        total_loss +=
            -target_logit +
            log_sum_exp;
    }

    // --------------------------------------------------------
    // Reduction.
    // --------------------------------------------------------

    float loss_value = 0.0f;

    if (reduction_mean_) {

        if (valid_count != 0) {

            loss_value =
                static_cast<float>(
                    total_loss /
                    static_cast<double>(
                        valid_count
                    )
                );
        }
        else {

            loss_value = 0.0f;
        }
    }
    else {

        loss_value =
            static_cast<float>(
                total_loss
            );
    }

    // --------------------------------------------------------
    // Scalar output.
    // --------------------------------------------------------

    Tensor result =
        Tensor::zeros(
            Shape{},
            DType::Float32,
            logits.device()
        );

    result.data_as<float>()[0] =
        loss_value;

    // --------------------------------------------------------
    // Autograd.
    // --------------------------------------------------------

    if (logits.requires_grad()) {

        result.set_grad_fn(
            make_cross_entropy_node(
                logits,
                targets,
                ignore_index_,
                reduction_mean_
            )
        );
    }

    return result;
}

// ============================================================
// METADATA
// ============================================================

std::int64_t
CrossEntropyLoss::ignore_index() const {
    return ignore_index_;
}

bool
CrossEntropyLoss::reduction_mean() const {
    return reduction_mean_;
}

} // namespace venla
