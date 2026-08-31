#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// KV CACHE
//
// Stores past Key / Value tensors for autoregressive
// Transformer inference.
//
// Supported cache layout:
//
//   [seq, embed_dim]
//
// and:
//
//   [batch, seq, embed_dim]
//
// The cache is CPU / Float32 for the initial implementation.
//
// Typical usage:
//
//   KVCache cache;
//
//   cache.append(key, value);
//
//   const Tensor& past_key = cache.key();
//   const Tensor& past_value = cache.value();
//
//   cache.clear();
//
// ============================================================

class KVCache {
public:

    KVCache();

    // --------------------------------------------------------
    // Append
    //
    // Appends new key/value tokens to the existing cache.
    //
    // First append:
    //
    //   [seq, embed]
    //
    // Subsequent append:
    //
    //   [new_seq, embed]
    //
    // The same layout must be used consistently.
    // --------------------------------------------------------

    void append(
        const Tensor& key,
        const Tensor& value
    );

    // --------------------------------------------------------
    // Clear
    // --------------------------------------------------------

    void clear();

    // --------------------------------------------------------
    // State
    // --------------------------------------------------------

    bool empty() const;

    std::size_t sequence_length() const;

    std::size_t embed_dim() const;

    std::size_t batch_size() const;

    // --------------------------------------------------------
    // Cached tensors
    // --------------------------------------------------------

    const Tensor& key() const;

    const Tensor& value() const;

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    bool initialized() const;

private:

    Tensor key_cache_;
    Tensor value_cache_;

    bool initialized_;

    std::size_t batch_size_;
    std::size_t sequence_length_;
    std::size_t embed_dim_;
};

} // namespace venla
