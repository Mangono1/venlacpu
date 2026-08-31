#include "venla/nn/kv_cache.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void validate_tensor(
    const Tensor& tensor,
    const char* name
) {
    if (tensor.dtype() != DType::Float32) {

        std::ostringstream message;

        message
            << "KVCache::append: "
            << name
            << " must be Float32";

        throw std::runtime_error(
            message.str()
        );
    }

    if (!tensor.device().is_cpu()) {

        std::ostringstream message;

        message
            << "KVCache::append: "
            << name
            << " must be on CPU";

        throw std::runtime_error(
            message.str()
        );
    }

    if (tensor.ndim() != 2 &&
        tensor.ndim() != 3) {

        std::ostringstream message;

        message
            << "KVCache::append: "
            << name
            << " must be 2D or 3D";

        throw std::runtime_error(
            message.str()
        );
    }
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

KVCache::KVCache()
    : key_cache_(),
      value_cache_(),
      initialized_(false),
      batch_size_(0),
      sequence_length_(0),
      embed_dim_(0) {
}

// ============================================================
// APPEND
//
// 2D:
//
//   cache:  [old_seq, embed]
//   input:  [new_seq, embed]
//   result: [old_seq + new_seq, embed]
//
// 3D:
//
//   cache:  [batch, old_seq, embed]
//   input:  [batch, new_seq, embed]
//   result: [batch, old_seq + new_seq, embed]
//
// ============================================================

void KVCache::append(
    const Tensor& key,
    const Tensor& value
) {
    validate_tensor(
        key,
        "key"
    );

    validate_tensor(
        value,
        "value"
    );

    // --------------------------------------------------------
    // Key and value must have identical shapes.
    // --------------------------------------------------------

    if (key.shape() != value.shape()) {

        throw std::runtime_error(
            "KVCache::append: "
            "key and value shapes must match"
        );
    }

    const std::size_t ndim =
        key.ndim();

    const std::size_t embed_dim =
        key.shape()[ndim - 1];

    const std::size_t new_sequence =
        key.shape()[ndim - 2];

    if (new_sequence == 0) {

        throw std::runtime_error(
            "KVCache::append: "
            "sequence length must be greater than zero"
        );
    }

    const std::size_t batch =
        ndim == 3
            ? key.shape()[0]
            : 1;

    // ========================================================
    // FIRST APPEND
    // ========================================================

    if (!initialized_) {

        key_cache_ =
            key;

        value_cache_ =
            value;

        batch_size_ =
            batch;

        sequence_length_ =
            new_sequence;

        embed_dim_ =
            embed_dim;

        initialized_ =
            true;

        return;
    }

    // ========================================================
    // VALIDATE EXISTING CACHE
    // ========================================================

    if (key.ndim() !=
        key_cache_.ndim()) {

        throw std::runtime_error(
            "KVCache::append: "
            "new tensors must use the same rank as the cache"
        );
    }

    if (batch != batch_size_) {

        throw std::runtime_error(
            "KVCache::append: "
            "batch size mismatch"
        );
    }

    if (embed_dim != embed_dim_) {

        throw std::runtime_error(
            "KVCache::append: "
            "embedding dimension mismatch"
        );
    }

    // ========================================================
    // 2D CACHE
    //
    // [sequence, embed]
    // ========================================================

    if (ndim == 2) {

        const std::size_t old_sequence =
            sequence_length_;

        const std::size_t total_sequence =
            old_sequence +
            new_sequence;

        Tensor new_key =
            Tensor::zeros(
                {
                    total_sequence,
                    embed_dim_
                },
                DType::Float32,
                key.device()
            );

        Tensor new_value =
            Tensor::zeros(
                {
                    total_sequence,
                    embed_dim_
                },
                DType::Float32,
                value.device()
            );

        const float* old_key =
            key_cache_.data_as<float>();

        const float* old_value =
            value_cache_.data_as<float>();

        const float* append_key =
            key.data_as<float>();

        const float* append_value =
            value.data_as<float>();

        float* destination_key =
            new_key.data_as<float>();

        float* destination_value =
            new_value.data_as<float>();

        const std::size_t old_elements =
            old_sequence *
            embed_dim_;

        const std::size_t append_elements =
            new_sequence *
            embed_dim_;

        // ----------------------------------------------------
        // Copy old sequence.
        // ----------------------------------------------------

        for (std::size_t i = 0;
             i < old_elements;
             ++i) {

            destination_key[i] =
                old_key[i];

            destination_value[i] =
                old_value[i];
        }

        // ----------------------------------------------------
        // Append new sequence.
        // ----------------------------------------------------

        for (std::size_t i = 0;
             i < append_elements;
             ++i) {

            destination_key[
                old_elements + i
            ] =
                append_key[i];

            destination_value[
                old_elements + i
            ] =
                append_value[i];
        }

        key_cache_ =
            std::move(
                new_key
            );

        value_cache_ =
            std::move(
                new_value
            );

        sequence_length_ =
            total_sequence;

        return;
    }

    // ========================================================
    // 3D CACHE
    //
    // [batch, sequence, embed]
    //
    // IMPORTANT:
    //
    // Each batch must be appended independently.
    //
    // Old:
    //
    //   B0: [old sequence]
    //   B1: [old sequence]
    //
    // New:
    //
    //   B0: [new sequence]
    //   B1: [new sequence]
    //
    // Result:
    //
    //   B0: [old + new]
    //   B1: [old + new]
    //
    // ========================================================

    const std::size_t old_sequence =
        sequence_length_;

    const std::size_t total_sequence =
        old_sequence +
        new_sequence;

    Tensor new_key =
        Tensor::zeros(
            {
                batch_size_,
                total_sequence,
                embed_dim_
            },
            DType::Float32,
            key.device()
        );

    Tensor new_value =
        Tensor::zeros(
            {
                batch_size_,
                total_sequence,
                embed_dim_
            },
            DType::Float32,
            value.device()
        );

    const float* old_key =
        key_cache_.data_as<float>();

    const float* old_value =
        value_cache_.data_as<float>();

    const float* append_key =
        key.data_as<float>();

    const float* append_value =
        value.data_as<float>();

    float* destination_key =
        new_key.data_as<float>();

    float* destination_value =
        new_value.data_as<float>();

    // Elements belonging to one batch in the old cache.
    const std::size_t old_batch_elements =
        old_sequence *
        embed_dim_;

    // Elements belonging to one batch in the new input.
    const std::size_t append_batch_elements =
        new_sequence *
        embed_dim_;

    // Elements belonging to one batch in the new cache.
    const std::size_t total_batch_elements =
        total_sequence *
        embed_dim_;

    // --------------------------------------------------------
    // Copy each batch independently.
    // --------------------------------------------------------

    for (std::size_t batch_index = 0;
         batch_index < batch_size_;
         ++batch_index) {

        const std::size_t old_offset =
            batch_index *
            old_batch_elements;

        const std::size_t append_offset =
            batch_index *
            append_batch_elements;

        const std::size_t destination_offset =
            batch_index *
            total_batch_elements;

        // ----------------------------------------------------
        // Existing tokens.
        // ----------------------------------------------------

        for (std::size_t i = 0;
             i < old_batch_elements;
             ++i) {

            destination_key[
                destination_offset + i
            ] =
                old_key[
                    old_offset + i
                ];

            destination_value[
                destination_offset + i
            ] =
                old_value[
                    old_offset + i
                ];
        }

        // ----------------------------------------------------
        // New tokens.
        // ----------------------------------------------------

        for (std::size_t i = 0;
             i < append_batch_elements;
             ++i) {

            destination_key[
                destination_offset +
                old_batch_elements +
                i
            ] =
                append_key[
                    append_offset + i
                ];

            destination_value[
                destination_offset +
                old_batch_elements +
                i
            ] =
                append_value[
                    append_offset + i
                ];
        }
    }

    // --------------------------------------------------------
    // Replace cache.
    // --------------------------------------------------------

    key_cache_ =
        std::move(
            new_key
        );

    value_cache_ =
        std::move(
            new_value
        );

    sequence_length_ =
        total_sequence;
}

// ============================================================
// CLEAR
// ============================================================

void KVCache::clear() {

    key_cache_ =
        Tensor();

    value_cache_ =
        Tensor();

    initialized_ =
        false;

    batch_size_ =
        0;

    sequence_length_ =
        0;

    embed_dim_ =
        0;
}

// ============================================================
// EMPTY
// ============================================================

bool KVCache::empty() const {
    return !initialized_;
}

// ============================================================
// SEQUENCE LENGTH
// ============================================================

std::size_t
KVCache::sequence_length() const {
    return sequence_length_;
}

// ============================================================
// EMBED DIM
// ============================================================

std::size_t
KVCache::embed_dim() const {
    return embed_dim_;
}

// ============================================================
// BATCH SIZE
// ============================================================

std::size_t
KVCache::batch_size() const {
    return batch_size_;
}

// ============================================================
// KEY
// ============================================================

const Tensor&
KVCache::key() const {

    if (!initialized_) {

        throw std::runtime_error(
            "KVCache::key: "
            "cache is empty"
        );
    }

    return key_cache_;
}

// ============================================================
// VALUE
// ============================================================

const Tensor&
KVCache::value() const {

    if (!initialized_) {

        throw std::runtime_error(
            "KVCache::value: "
            "cache is empty"
        );
    }

    return value_cache_;
}

// ============================================================
// INITIALIZED
// ============================================================

bool KVCache::initialized() const {
    return initialized_;
}

} // namespace venla
