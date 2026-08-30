#pragma once

#include "venla/tokenizer/vocabulary.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace venla {

// ============================================================
// BPE TOKENIZER
//
// Subword tokenizer untuk fondasi NLP / LLM.
//
// Pipeline:
//
//     text
//       ↓
//     UTF-8 codepoints
//       ↓
//     BPE merges
//       ↓
//     token IDs
//
// Training:
//
//     corpus
//       ↓
//     frequency counting
//       ↓
//     most frequent pair
//       ↓
//     merge
//       ↓
//     vocabulary
//
// ============================================================

class BPETokenizer {
public:

    struct Config {
        std::size_t vocab_size = 30000;

        std::size_t min_frequency = 2;

        bool add_bos = false;

        bool add_eos = false;
    };

    BPETokenizer();

    explicit BPETokenizer(
        const Config& config
    );

    // --------------------------------------------------------
    // TRAINING
    // --------------------------------------------------------

    void train(
        const std::string& corpus
    );

    void train(
        const std::vector<std::string>& documents
    );

    // --------------------------------------------------------
    // ENCODE
    // --------------------------------------------------------

    std::vector<std::size_t> encode(
        const std::string& text
    ) const;

    std::vector<std::size_t> encode(
        const std::string& text,
        bool add_bos,
        bool add_eos
    ) const;

    // --------------------------------------------------------
    // DECODE
    // --------------------------------------------------------

    std::string decode(
        const std::vector<std::size_t>& ids
    ) const;

    // --------------------------------------------------------
    // VOCABULARY
    // --------------------------------------------------------

    const Vocabulary& vocabulary() const;

    Vocabulary& vocabulary();

    // --------------------------------------------------------
    // CONFIG
    // --------------------------------------------------------

    const Config& config() const;

    // --------------------------------------------------------
    // TRAINING STATE
    // --------------------------------------------------------

    bool trained() const;

    std::size_t merge_count() const;

    // --------------------------------------------------------
    // PERSISTENCE
    // --------------------------------------------------------

    void save(
        const std::string& path
    ) const;

    static BPETokenizer load(
        const std::string& path
    );

private:

    struct MergeRule {
        std::string left;
        std::string right;
        std::size_t token_id;
    };

    struct PairKey {
        std::string left;
        std::string right;

        bool operator==(
            const PairKey& other
        ) const;
    };

    struct PairKeyHash {
        std::size_t operator()(
            const PairKey& key
        ) const;
    };

    using SymbolSequence =
        std::vector<std::string>;

    Config config_;

    Vocabulary vocabulary_;

    std::vector<MergeRule> merges_;

    std::unordered_map<
        PairKey,
        std::size_t,
        PairKeyHash
    > merge_ranks_;

    bool trained_;

    // --------------------------------------------------------
    // UTF-8
    // --------------------------------------------------------

    static SymbolSequence split_utf8(
        const std::string& text
    );

    static bool valid_utf8(
        const std::string& text
    );

    // --------------------------------------------------------
    // BPE INTERNAL
    // --------------------------------------------------------

    static std::unordered_map<
        PairKey,
        std::size_t,
        PairKeyHash
    > count_pairs(
        const std::vector<
            SymbolSequence
        >& sequences
    );

    static void merge_pair(
        std::vector<
            SymbolSequence
        >& sequences,
        const PairKey& pair,
        const std::string& merged
    );

    SymbolSequence apply_merges(
        const std::string& text
    ) const;

    void rebuild_merge_ranks();
};

} // namespace venla
