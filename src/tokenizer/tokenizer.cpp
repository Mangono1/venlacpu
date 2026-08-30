#include "venla/tokenizer/tokenizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace venla {

namespace {

std::size_t hash_string(
    const std::string& value
) {
    return std::hash<std::string>{}(
        value
    );
}

} // namespace

// ============================================================
// PAIR KEY
// ============================================================

bool BPETokenizer::PairKey::operator==(
    const PairKey& other
) const {
    return left == other.left &&
           right == other.right;
}

std::size_t
BPETokenizer::PairKeyHash::operator()(
    const PairKey& key
) const {

    const std::size_t h1 =
        hash_string(key.left);

    const std::size_t h2 =
        hash_string(key.right);

    return h1 ^
        (
            h2 +
            static_cast<std::size_t>(
                0x9e3779b97f4a7c15ULL
            ) +
            (h1 << 6) +
            (h1 >> 2)
        );
}

// ============================================================
// CONSTRUCTORS
// ============================================================

BPETokenizer::BPETokenizer()
    : config_(),
      vocabulary_(),
      merges_(),
      merge_ranks_(),
      trained_(false) {
}

BPETokenizer::BPETokenizer(
    const Config& config
)
    : config_(config),
      vocabulary_(),
      merges_(),
      merge_ranks_(),
      trained_(false) {

    if (config_.vocab_size <
        vocabulary_.size()) {

        throw std::invalid_argument(
            "BPETokenizer: "
            "vocab_size is smaller than "
            "special token count"
        );
    }

    if (config_.min_frequency == 0) {
        throw std::invalid_argument(
            "BPETokenizer: "
            "min_frequency must be greater than zero"
        );
    }
}

// ============================================================
// UTF-8 SPLITTER
//
// Mengambil Unicode codepoint sebagai unit awal BPE.
//
// Ini penting untuk Bahasa Indonesia dan teks multilingual.
// BPE kemudian menggabungkan codepoint menjadi subword.
//
// Invalid UTF-8 dilempar sebagai error, bukan diam-diam rusak.
// ============================================================

BPETokenizer::SymbolSequence
BPETokenizer::split_utf8(
    const std::string& text
) {
    SymbolSequence result;

    std::size_t i = 0;

    while (i < text.size()) {

        const unsigned char first =
            static_cast<unsigned char>(
                text[i]
            );

        std::size_t length = 0;

        if (first < 0x80) {
            length = 1;
        }
        else if (
            (first & 0xE0) == 0xC0
        ) {
            length = 2;
        }
        else if (
            (first & 0xF0) == 0xE0
        ) {
            length = 3;
        }
        else if (
            (first & 0xF8) == 0xF0
        ) {
            length = 4;
        }
        else {
            throw std::runtime_error(
                "BPETokenizer: invalid UTF-8 leading byte"
            );
        }

        if (i + length > text.size()) {
            throw std::runtime_error(
                "BPETokenizer: truncated UTF-8 sequence"
            );
        }

        for (std::size_t j = 1;
             j < length;
             ++j) {

            const unsigned char continuation =
                static_cast<unsigned char>(
                    text[i + j]
                );

            if (
                (continuation & 0xC0)
                != 0x80
            ) {
                throw std::runtime_error(
                    "BPETokenizer: invalid UTF-8 continuation byte"
                );
            }
        }

        result.push_back(
            text.substr(
                i,
                length
            )
        );

        i += length;
    }

    return result;
}

// ============================================================
// UTF-8 VALIDATION
// ============================================================

bool BPETokenizer::valid_utf8(
    const std::string& text
) {
    try {
        split_utf8(text);
        return true;
    }
    catch (...) {
        return false;
    }
}

// ============================================================
// COUNT PAIRS
// ============================================================

std::unordered_map<
    BPETokenizer::PairKey,
    std::size_t,
    BPETokenizer::PairKeyHash
>
BPETokenizer::count_pairs(
    const std::vector<SymbolSequence>& sequences
) {

    std::unordered_map<
        PairKey,
        std::size_t,
        PairKeyHash
    > counts;

    for (const SymbolSequence& sequence :
         sequences) {

        if (sequence.size() < 2) {
            continue;
        }

        for (std::size_t i = 0;
             i + 1 < sequence.size();
             ++i) {

            PairKey key{
                sequence[i],
                sequence[i + 1]
            };

            ++counts[key];
        }
    }

    return counts;
}

// ============================================================
// MERGE PAIR
// ============================================================

void BPETokenizer::merge_pair(
    std::vector<SymbolSequence>& sequences,
    const PairKey& pair,
    const std::string& merged
) {

    for (SymbolSequence& sequence :
         sequences) {

        if (sequence.size() < 2) {
            continue;
        }

        SymbolSequence result;

        result.reserve(
            sequence.size()
        );

        std::size_t i = 0;

        while (i < sequence.size()) {

            if (
                i + 1 < sequence.size() &&
                sequence[i] == pair.left &&
                sequence[i + 1] == pair.right
            ) {

                result.push_back(
                    merged
                );

                i += 2;
            }
            else {

                result.push_back(
                    sequence[i]
                );

                ++i;
            }
        }

        sequence =
            std::move(result);
    }
}

// ============================================================
// TRAIN
// ============================================================

void BPETokenizer::train(
    const std::string& corpus
) {
    std::vector<std::string> documents;

    documents.push_back(
        corpus
    );

    train(documents);
}

// ============================================================
// TRAIN DOCUMENTS
// ============================================================

void BPETokenizer::train(
    const std::vector<std::string>& documents
) {
    if (documents.empty()) {
        throw std::invalid_argument(
            "BPETokenizer::train: "
            "documents cannot be empty"
        );
    }

    // --------------------------------------------------------
    // Reset
    // --------------------------------------------------------

    vocabulary_ =
        Vocabulary();

    merges_.clear();

    merge_ranks_.clear();

    trained_ =
        false;

    // --------------------------------------------------------
    // Convert documents to UTF-8 symbol sequences
    // --------------------------------------------------------

    std::vector<SymbolSequence>
        sequences;

    sequences.reserve(
        documents.size()
    );

    // --------------------------------------------------------
    // Collect base symbols
    // --------------------------------------------------------

    for (const std::string& document :
         documents) {

        if (!valid_utf8(document)) {
            throw std::runtime_error(
                "BPETokenizer::train: "
                "document contains invalid UTF-8"
            );
        }

        SymbolSequence sequence =
            split_utf8(document);

        sequences.push_back(
            sequence
        );

        for (const std::string& symbol :
             sequence) {

            vocabulary_.add_token(
                symbol
            );
        }
    }

    if (vocabulary_.size() >
        config_.vocab_size) {

        throw std::runtime_error(
            "BPETokenizer::train: "
            "base vocabulary already exceeds vocab_size"
        );
    }

    // --------------------------------------------------------
    // BPE merge loop
    // --------------------------------------------------------

    while (
        vocabulary_.size()
        <
        config_.vocab_size
    ) {

        const auto pair_counts =
            count_pairs(
                sequences
            );

        if (pair_counts.empty()) {
            break;
        }

        // Deterministic selection:
        //
        // 1. highest frequency
        // 2. lexical left token
        // 3. lexical right token
        //
        // Dengan ini training tidak bergantung pada urutan
        // hash table.
        //

        bool found =
            false;

        PairKey best_pair;

        std::size_t best_count =
            0;

        for (
            const auto& entry :
            pair_counts
        ) {

            const PairKey& pair =
                entry.first;

            const std::size_t count =
                entry.second;

            if (
                count <
                config_.min_frequency
            ) {
                continue;
            }

            if (!found) {

                found = true;

                best_pair =
                    pair;

                best_count =
                    count;

                continue;
            }

            if (count > best_count) {

                best_pair =
                    pair;

                best_count =
                    count;

                continue;
            }

            if (
                count == best_count &&
                (
                    pair.left <
                    best_pair.left ||
                    (
                        pair.left ==
                        best_pair.left &&
                        pair.right <
                        best_pair.right
                    )
                )
            ) {

                best_pair =
                    pair;
            }
        }

        if (!found) {
            break;
        }

        const std::string merged =
            best_pair.left +
            best_pair.right;

        const std::size_t token_id =
            vocabulary_.add_token(
                merged
            );

        merges_.push_back(
            MergeRule{
                best_pair.left,
                best_pair.right,
                token_id
            }
        );

        merge_pair(
            sequences,
            best_pair,
            merged
        );
    }

    rebuild_merge_ranks();

    trained_ =
        true;
}

// ============================================================
// REBUILD MERGE RANKS
// ============================================================

void BPETokenizer::rebuild_merge_ranks() {

    merge_ranks_.clear();

    for (std::size_t i = 0;
         i < merges_.size();
         ++i) {

        const MergeRule& merge =
            merges_[i];

        merge_ranks_.emplace(
            PairKey{
                merge.left,
                merge.right
            },
            i
        );
    }
}

// ============================================================
// APPLY MERGES
// ============================================================

BPETokenizer::SymbolSequence
BPETokenizer::apply_merges(
    const std::string& text
) const {

    SymbolSequence sequence =
        split_utf8(text);

    if (sequence.size() < 2) {
        return sequence;
    }

    // Apply rules in learned rank order.
    for (const MergeRule& merge :
         merges_) {

        if (sequence.size() < 2) {
            break;
        }

        SymbolSequence result;

        result.reserve(
            sequence.size()
        );

        std::size_t i = 0;

        while (
            i < sequence.size()
        ) {

            if (
                i + 1 < sequence.size() &&
                sequence[i] ==
                    merge.left &&
                sequence[i + 1] ==
                    merge.right
            ) {

                result.push_back(
                    merge.left +
                    merge.right
                );

                i += 2;
            }
            else {

                result.push_back(
                    sequence[i]
                );

                ++i;
            }
        }

        sequence =
            std::move(result);
    }

    return sequence;
}

// ============================================================
// ENCODE
// ============================================================

std::vector<std::size_t>
BPETokenizer::encode(
    const std::string& text
) const {
    return encode(
        text,
        config_.add_bos,
        config_.add_eos
    );
}

// ============================================================
// ENCODE WITH SPECIAL TOKENS
// ============================================================

std::vector<std::size_t>
BPETokenizer::encode(
    const std::string& text,
    bool add_bos,
    bool add_eos
) const {

    if (!trained_) {
        throw std::runtime_error(
            "BPETokenizer::encode: "
            "tokenizer has not been trained"
        );
    }

    if (!valid_utf8(text)) {
        throw std::runtime_error(
            "BPETokenizer::encode: "
            "text contains invalid UTF-8"
        );
    }

    const SymbolSequence symbols =
        apply_merges(text);

    std::vector<std::size_t> ids;

    ids.reserve(
        symbols.size() +
        (add_bos ? 1 : 0) +
        (add_eos ? 1 : 0)
    );

    if (add_bos) {
        ids.push_back(
            vocabulary_.bos_id()
        );
    }

    for (const std::string& symbol :
         symbols) {

        ids.push_back(
            vocabulary_.token_to_id(
                symbol
            )
        );
    }

    if (add_eos) {
        ids.push_back(
            vocabulary_.eos_id()
        );
    }

    return ids;
}

// ============================================================
// DECODE
// ============================================================

std::string BPETokenizer::decode(
    const std::vector<std::size_t>& ids
) const {

    if (!trained_) {
        throw std::runtime_error(
            "BPETokenizer::decode: "
            "tokenizer has not been trained"
        );
    }

    std::string result;

    for (const std::size_t id : ids) {

        if (!vocabulary_.contains_id(id)) {
            result +=
                vocabulary_.unk_token();

            continue;
        }

        if (id == vocabulary_.pad_id() ||
            id == vocabulary_.bos_id() ||
            id == vocabulary_.eos_id()) {

            continue;
        }

        result +=
            vocabulary_.id_to_token(id);
    }

    return result;
}

// ============================================================
// VOCABULARY
// ============================================================

const Vocabulary&
BPETokenizer::vocabulary() const {
    return vocabulary_;
}

Vocabulary&
BPETokenizer::vocabulary() {
    return vocabulary_;
}

// ============================================================
// CONFIG
// ============================================================

const BPETokenizer::Config&
BPETokenizer::config() const {
    return config_;
}

// ============================================================
// TRAINED
// ============================================================

bool BPETokenizer::trained() const {
    return trained_;
}

// ============================================================
// MERGE COUNT
// ============================================================

std::size_t
BPETokenizer::merge_count() const {
    return merges_.size();
}

// ============================================================
// SAVE
//
// Format:
//
// VENLA_BPE_V1
// CONFIG ...
// VOCAB ...
// MERGES ...
//
// ============================================================

void BPETokenizer::save(
    const std::string& path
) const {

    if (!trained_) {
        throw std::runtime_error(
            "BPETokenizer::save: "
            "tokenizer has not been trained"
        );
    }

    const std::string vocab_path =
        path + ".vocab";

    vocabulary_.save(
        vocab_path
    );

    std::ofstream output(
        path,
        std::ios::binary
    );

    if (!output) {
        throw std::runtime_error(
            "BPETokenizer::save: "
            "cannot open file"
        );
    }

    output
        << "VENLA_BPE_V1\n";

    output
        << "VOCAB_FILE "
        << vocab_path.size()
        << " "
        << vocab_path
        << "\n";

    output
        << "VOCAB_SIZE "
        << config_.vocab_size
        << "\n";

    output
        << "MIN_FREQUENCY "
        << config_.min_frequency
        << "\n";

    output
        << "ADD_BOS "
        << (
            config_.add_bos
                ? 1
                : 0
        )
        << "\n";

    output
        << "ADD_EOS "
        << (
            config_.add_eos
                ? 1
                : 0
        )
        << "\n";

    output
        << "MERGES "
        << merges_.size()
        << "\n";

    for (const MergeRule& merge :
         merges_) {

        output
            << merge.left.size()
            << " "
            << merge.left
            << " "
            << merge.right.size()
            << " "
            << merge.right
            << " "
            << merge.token_id
            << "\n";
    }

    if (!output) {
        throw std::runtime_error(
            "BPETokenizer::save: "
            "write failure"
        );
    }
}

// ============================================================
// LOAD
// ============================================================

BPETokenizer BPETokenizer::load(
    const std::string& path
) {

    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        throw std::runtime_error(
            "BPETokenizer::load: "
            "cannot open file"
        );
    }

    std::string magic;

    std::getline(
        input,
        magic
    );

    if (magic != "VENLA_BPE_V1") {
        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid tokenizer format"
        );
    }

    std::string label;

    std::size_t vocab_path_length;

    input
        >> label
        >> vocab_path_length;

    if (!input ||
        label != "VOCAB_FILE") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid VOCAB_FILE"
        );
    }

    char separator;

    input.get(separator);

    if (separator != ' ') {
        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid vocabulary path"
        );
    }

    std::string vocab_path(
        vocab_path_length,
        '\0'
    );

    input.read(
        &vocab_path[0],
        static_cast<std::streamsize>(
            vocab_path_length
        )
    );

    input.get(separator);

    if (!input ||
        separator != '\n') {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid vocabulary path record"
        );
    }

    Config config;

    input
        >> label
        >> config.vocab_size;

    if (!input ||
        label != "VOCAB_SIZE") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid VOCAB_SIZE"
        );
    }

    input
        >> label
        >> config.min_frequency;

    if (!input ||
        label != "MIN_FREQUENCY") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid MIN_FREQUENCY"
        );
    }

    int add_bos;

    input
        >> label
        >> add_bos;

    if (!input ||
        label != "ADD_BOS") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid ADD_BOS"
        );
    }

    int add_eos;

    input
        >> label
        >> add_eos;

    if (!input ||
        label != "ADD_EOS") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid ADD_EOS"
        );
    }

    config.add_bos =
        add_bos != 0;

    config.add_eos =
        add_eos != 0;

    BPETokenizer tokenizer(
        config
    );

    tokenizer.vocabulary_ =
        Vocabulary::load(
            vocab_path
        );

    std::size_t merge_count;

    input
        >> label
        >> merge_count;

    if (!input ||
        label != "MERGES") {

        throw std::runtime_error(
            "BPETokenizer::load: "
            "invalid MERGES section"
        );
    }

    input.ignore(
        std::numeric_limits<
            std::streamsize
        >::max(),
        '\n'
    );

    tokenizer.merges_.clear();

    for (std::size_t i = 0;
         i < merge_count;
         ++i) {

        std::size_t left_length;

        std::size_t right_length;

        std::size_t token_id;

        input
            >> left_length;

        input.get(separator);

        if (!input ||
            separator != ' ') {

            throw std::runtime_error(
                "BPETokenizer::load: "
                "invalid left token"
            );
        }

        std::string left(
            left_length,
            '\0'
        );

        input.read(
            &left[0],
            static_cast<std::streamsize>(
                left_length
            )
        );

        input.get(separator);

        if (!input ||
            separator != ' ') {

            throw std::runtime_error(
                "BPETokenizer::load: "
                "invalid left token separator"
            );
        }

        input
            >> right_length;

        input.get(separator);

        if (!input ||
            separator != ' ') {

            throw std::runtime_error(
                "BPETokenizer::load: "
                "invalid right token"
            );
        }

        std::string right(
            right_length,
            '\0'
        );

        input.read(
            &right[0],
            static_cast<std::streamsize>(
                right_length
            )
        );

        input
            >> token_id;

        input.get(separator);

        if (!input ||
            separator != '\n') {

            throw std::runtime_error(
                "BPETokenizer::load: "
                "invalid merge record"
            );
        }

        tokenizer.merges_.push_back(
            MergeRule{
                left,
                right,
                token_id
            }
        );
    }

    tokenizer.rebuild_merge_ranks();

    tokenizer.trained_ =
        true;

    return tokenizer;
}

} // namespace venla
