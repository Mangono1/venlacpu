#pragma once

#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace venla {

// ============================================================
// CHECKPOINT METADATA
// ============================================================

struct CheckpointMetadata {

    std::size_t epoch = 0;

    std::size_t global_step = 0;

    float best_eval_loss = 0.0f;

    std::size_t bad_epochs = 0;

    bool has_best_model = false;
};

// ============================================================
// CHECKPOINT ENGINE
//
// Binary checkpoint:
//
//     model architecture
//     model parameters
//     optimizer state
//     trainer state
//
// Format versioned dengan magic:
//     VNLCP230
// ============================================================

class CheckpointEngine {
public:

    static void save(
        const std::string& path,
        const LanguageModel& model,
        const Optimizer& optimizer,
        const CheckpointMetadata& metadata
    );

    static CheckpointMetadata load(
        const std::string& path,
        LanguageModel& model,
        Optimizer& optimizer
    );

};

} // namespace venla
