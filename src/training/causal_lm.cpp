#include "venla/training/causal_lm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace venla {

namespace {

void validate_token_tensor(
    const Tensor& tokens,
    const char* operation
) {
    if (tokens.ndim() != 1 &&
        tokens.ndim() != 2) {

        throw std::runtime_error(
            std::string(operation) +
            ": tokens must be 1D or 2D"
        );
    }

    if (tokens.dtype() != DType::Int32 &&
        tokens.dtype() != DType::Int64) {

        throw std::runtime_error(
            std::string(operation) +
            ": tokens must be Int32 or Int64"
        );
    }

    if (!tokens.device().is_cpu()) {

        throw std::runtime_error(
            std::string(operation) +
            ": tokens must be on CPU"
        );
    }

    if (tokens.numel() == 0) {

        throw std::runtime_error(
            std::string(operation) +
            ": tokens cannot be empty"
        );
    }
}

} // namespace

CausalLMDataset::CausalLMDataset(
    std::int64_t pad_token_id,
    std::int64_t ignore_index
)
    : sequences_(),
      pad_token_id_(pad_token_id),
      ignore_index_(ignore_index) {

    if (pad_token_id == ignore_index) {

        throw std::invalid_argument(
            "CausalLMDataset: "
            "pad_token_id and ignore_index "
            "must be different"
        );
    }
}

void CausalLMDataset::add_sequence(
    const std::vector<std::int64_t>& tokens
) {
    if (tokens.size() < 2) {

        throw std::invalid_argument(
            "CausalLMDataset::add_sequence: "
            "sequence must contain at least two tokens"
        );
    }

    sequences_.push_back(tokens);
}

void CausalLMDataset::add_sequence_int32(
    const std::vector<std::int32_t>& tokens
) {
    if (tokens.size() < 2) {
        throw std::invalid_argument(
            "CausalLMDataset::add_sequence_int32: sequence must contain at least two tokens"
        );
    }

    std::vector<std::int64_t> converted;
    converted.reserve(tokens.size());

    for (std::int32_t token : tokens) {
        converted.push_back(
            static_cast<std::int64_t>(token)
        );
    }

    sequences_.push_back(
        std::move(converted)
    );
}

void CausalLMDataset::clear() {
    sequences_.clear();
}

std::size_t CausalLMDataset::size() const {
    return sequences_.size();
}

bool CausalLMDataset::empty() const {
    return sequences_.empty();
}

std::size_t
CausalLMDataset::max_sequence_length() const {

    std::size_t maximum = 0;

    for (const auto& sequence : sequences_) {

        maximum =
            std::max(
                maximum,
                sequence.size()
            );
    }

    return maximum;
}

std::size_t CausalLMDataset::num_batches(
    std::size_t batch_size,
    bool drop_last
) const {

    if (batch_size == 0) {

        throw std::invalid_argument(
            "CausalLMDataset::num_batches: "
            "batch_size must be greater than zero"
        );
    }

    if (sequences_.empty()) {
        return 0;
    }

    if (drop_last) {
        return sequences_.size() / batch_size;
    }

    return (
        sequences_.size() +
        batch_size -
        1
    ) / batch_size;
}

CausalLMBatch CausalLMDataset::batch(
    std::size_t batch_index,
    std::size_t batch_size,
    bool drop_last
) const {

    const std::size_t total_batches =
        num_batches(
            batch_size,
            drop_last
        );

    if (batch_index >= total_batches) {

        throw std::out_of_range(
            "CausalLMDataset::batch: "
            "batch index out of range"
        );
    }

    const std::size_t start =
        batch_index * batch_size;

    const std::size_t remaining =
        sequences_.size() - start;

    const std::size_t actual_batch =
        std::min(
            batch_size,
            remaining
        );

    if (drop_last &&
        actual_batch != batch_size) {

        throw std::runtime_error(
            "CausalLMDataset::batch: "
            "incomplete batch"
        );
    }

    std::size_t maximum_length = 0;

    for (std::size_t i = 0;
         i < actual_batch;
         ++i) {

        maximum_length =
            std::max(
                maximum_length,
                sequences_[start + i].size()
            );
    }

    if (maximum_length < 2) {

        throw std::runtime_error(
            "CausalLMDataset::batch: "
            "sequence must contain at least two tokens"
        );
    }

    const std::size_t shifted_length =
        maximum_length - 1;

    Tensor input =
        Tensor::empty(
            {
                actual_batch,
                shifted_length
            },
            DType::Int64,
            Device::cpu()
        );

    Tensor targets =
        Tensor::empty(
            {
                actual_batch,
                shifted_length
            },
            DType::Int64,
            Device::cpu()
        );

    std::int64_t* input_data =
        input.data_as<std::int64_t>();

    std::int64_t* target_data =
        targets.data_as<std::int64_t>();

    std::size_t valid_tokens = 0;

    for (std::size_t row = 0;
         row < actual_batch;
         ++row) {

        const auto& sequence =
            sequences_[start + row];

        for (std::size_t position = 0;
             position < shifted_length;
             ++position) {

            const std::size_t destination =
                row *
                shifted_length +
                position;

            if (position < sequence.size() - 1) {

                input_data[destination] =
                    sequence[position];

                target_data[destination] =
                    sequence[position + 1];

                ++valid_tokens;
            }
            else {

                input_data[destination] =
                    pad_token_id_;

                target_data[destination] =
                    ignore_index_;
            }
        }
    }

    CausalLMBatch result;

    result.input =
        std::move(input);

    result.targets =
        std::move(targets);

    result.batch_size =
        actual_batch;

    result.sequence_length =
        shifted_length;

    result.valid_tokens =
        valid_tokens;

    return result;
}

std::int64_t
CausalLMDataset::pad_token_id() const {
    return pad_token_id_;
}

std::int64_t
CausalLMDataset::ignore_index() const {
    return ignore_index_;
}

CausalLMBatch make_causal_lm_batch(
    const Tensor& tokens,
    std::int64_t ignore_index
) {
    validate_token_tensor(
        tokens,
        "make_causal_lm_batch"
    );

    std::size_t batch_size = 1;
    std::size_t sequence_length = 0;

    if (tokens.ndim() == 1) {

        sequence_length =
            tokens.shape()[0];
    }
    else {

        batch_size =
            tokens.shape()[0];

        sequence_length =
            tokens.shape()[1];
    }

    if (sequence_length < 2) {

        throw std::invalid_argument(
            "make_causal_lm_batch: "
            "sequence length must be at least 2"
        );
    }

    const std::size_t shifted_length =
        sequence_length - 1;

    Shape shifted_shape =
        tokens.ndim() == 1
            ? Shape({shifted_length})
            : Shape({
                batch_size,
                shifted_length
            });

    Tensor input =
        Tensor::empty(
            shifted_shape,
            tokens.dtype(),
            Device::cpu()
        );

    Tensor targets =
        Tensor::empty(
            shifted_shape,
            tokens.dtype(),
            Device::cpu()
        );

    for (std::size_t row = 0;
         row < batch_size;
         ++row) {

        for (std::size_t position = 0;
             position < shifted_length;
             ++position) {

            const std::size_t source =
                row *
                sequence_length +
                position;

            const std::size_t destination =
                row *
                shifted_length +
                position;

            if (tokens.dtype() == DType::Int32) {

                const std::int32_t* source_data =
                    tokens.data_as<std::int32_t>();

                std::int32_t* input_data =
                    input.data_as<std::int32_t>();

                std::int32_t* target_data =
                    targets.data_as<std::int32_t>();

                input_data[destination] =
                    source_data[source];

                target_data[destination] =
                    source_data[source + 1];
            }
            else {

                const std::int64_t* source_data =
                    tokens.data_as<std::int64_t>();

                std::int64_t* input_data =
                    input.data_as<std::int64_t>();

                std::int64_t* target_data =
                    targets.data_as<std::int64_t>();

                input_data[destination] =
                    source_data[source];

                target_data[destination] =
                    source_data[source + 1];
            }
        }
    }

    CausalLMBatch result;

    result.input =
        std::move(input);

    result.targets =
        std::move(targets);

    result.batch_size =
        batch_size;

    result.sequence_length =
        shifted_length;

    result.valid_tokens =
        batch_size *
        shifted_length;

    return result;
}

} // namespace venla
