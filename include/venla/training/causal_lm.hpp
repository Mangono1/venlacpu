#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace venla {

struct CausalLMBatch {
    Tensor input;
    Tensor targets;

    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    std::size_t valid_tokens = 0;
};

class CausalLMDataset {
public:

    explicit CausalLMDataset(
        std::int64_t pad_token_id = 0,
        std::int64_t ignore_index = -100
    );

    void add_sequence(
        const std::vector<std::int64_t>& tokens
    );

    void add_sequence_int32(
        const std::vector<std::int32_t>& tokens
    );

    void clear();

    std::size_t size() const;
    bool empty() const;
    std::size_t max_sequence_length() const;

    std::size_t num_batches(
        std::size_t batch_size,
        bool drop_last = false
    ) const;

    CausalLMBatch batch(
        std::size_t batch_index,
        std::size_t batch_size,
        bool drop_last = false
    ) const;

    std::int64_t pad_token_id() const;
    std::int64_t ignore_index() const;

private:

    std::vector<std::vector<std::int64_t>> sequences_;
    std::int64_t pad_token_id_;
    std::int64_t ignore_index_;
};

CausalLMBatch make_causal_lm_batch(
    const Tensor& tokens,
    std::int64_t ignore_index = -100
);

} // namespace venla

