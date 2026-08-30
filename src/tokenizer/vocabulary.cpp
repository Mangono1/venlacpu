#include "venla/tokenizer/vocabulary.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

Vocabulary::Vocabulary()
    : tokens_(),
      token_to_id_(),
      pad_token_("<PAD>"),
      unk_token_("<UNK>"),
      bos_token_("<BOS>"),
      eos_token_("<EOS>"),
      pad_id_(0),
      unk_id_(0),
      bos_id_(0),
      eos_id_(0) {

    pad_id_ =
        add_token(pad_token_);

    unk_id_ =
        add_token(unk_token_);

    bos_id_ =
        add_token(bos_token_);

    eos_id_ =
        add_token(eos_token_);
}

// ============================================================
// ADD TOKEN
// ============================================================

std::size_t Vocabulary::add_token(
    const std::string& token
) {
    if (token.empty()) {
        throw std::invalid_argument(
            "Vocabulary::add_token: "
            "token cannot be empty"
        );
    }

    auto iterator =
        token_to_id_.find(token);

    if (iterator != token_to_id_.end()) {
        return iterator->second;
    }

    const std::size_t id =
        tokens_.size();

    tokens_.push_back(token);

    token_to_id_.emplace(
        token,
        id
    );

    return id;
}

// ============================================================
// CONTAINS
// ============================================================

bool Vocabulary::contains(
    const std::string& token
) const {
    return token_to_id_.find(token)
        != token_to_id_.end();
}

bool Vocabulary::contains_id(
    std::size_t id
) const {
    return id < tokens_.size();
}

// ============================================================
// TOKEN -> ID
// ============================================================

std::size_t Vocabulary::token_to_id(
    const std::string& token
) const {
    auto iterator =
        token_to_id_.find(token);

    if (iterator == token_to_id_.end()) {
        return unk_id_;
    }

    return iterator->second;
}

// ============================================================
// ID -> TOKEN
// ============================================================

const std::string&
Vocabulary::id_to_token(
    std::size_t id
) const {
    if (id >= tokens_.size()) {
        throw std::out_of_range(
            "Vocabulary::id_to_token: "
            "id out of range"
        );
    }

    return tokens_[id];
}

// ============================================================
// SIZE
// ============================================================

std::size_t Vocabulary::size() const {
    return tokens_.size();
}

bool Vocabulary::empty() const {
    return tokens_.empty();
}

// ============================================================
// SPECIAL IDS
// ============================================================

std::size_t Vocabulary::pad_id() const {
    return pad_id_;
}

std::size_t Vocabulary::unk_id() const {
    return unk_id_;
}

std::size_t Vocabulary::bos_id() const {
    return bos_id_;
}

std::size_t Vocabulary::eos_id() const {
    return eos_id_;
}

// ============================================================
// SPECIAL TOKENS
// ============================================================

const std::string&
Vocabulary::pad_token() const {
    return pad_token_;
}

const std::string&
Vocabulary::unk_token() const {
    return unk_token_;
}

const std::string&
Vocabulary::bos_token() const {
    return bos_token_;
}

const std::string&
Vocabulary::eos_token() const {
    return eos_token_;
}

// ============================================================
// SET SPECIAL TOKENS
//
// Special token strings are configurable, but their IDs remain
// stable once assigned by the constructor.
//
// This method is primarily intended for persistence/loading
// compatibility.
// ============================================================

void Vocabulary::set_special_tokens(
    const std::string& pad,
    const std::string& unk,
    const std::string& bos,
    const std::string& eos
) {
    if (pad.empty() ||
        unk.empty() ||
        bos.empty() ||
        eos.empty()) {

        throw std::invalid_argument(
            "Vocabulary::set_special_tokens: "
            "special tokens cannot be empty"
        );
    }

    pad_token_ = pad;
    unk_token_ = unk;
    bos_token_ = bos;
    eos_token_ = eos;

    pad_id_ =
        token_to_id(pad_token_);

    unk_id_ =
        token_to_id(unk_token_);

    bos_id_ =
        token_to_id(bos_token_);

    eos_id_ =
        token_to_id(eos_token_);
}

// ============================================================
// TOKENS
// ============================================================

const std::vector<std::string>&
Vocabulary::tokens() const {
    return tokens_;
}

// ============================================================
// SAVE
//
// Binary-free textual format:
//
// VENLA_VOCAB_V1
// PAD <length> <token>
// UNK <length> <token>
// BOS <length> <token>
// EOS <length> <token>
// TOKENS <count>
// <length> <token>
// ...
//
// Length makes the format safe for spaces/newlines inside
// tokens.
// ============================================================

void Vocabulary::save(
    const std::string& path
) const {

    std::ofstream output(
        path,
        std::ios::binary
    );

    if (!output) {
        throw std::runtime_error(
            "Vocabulary::save: "
            "cannot open file"
        );
    }

    output
        << "VENLA_VOCAB_V1\n";

    output
        << "PAD "
        << pad_token_.size()
        << " "
        << pad_token_
        << "\n";

    output
        << "UNK "
        << unk_token_.size()
        << " "
        << unk_token_
        << "\n";

    output
        << "BOS "
        << bos_token_.size()
        << " "
        << bos_token_
        << "\n";

    output
        << "EOS "
        << eos_token_.size()
        << " "
        << eos_token_
        << "\n";

    output
        << "TOKENS "
        << tokens_.size()
        << "\n";

    for (const std::string& token : tokens_) {

        output
            << token.size()
            << " "
            << token
            << "\n";
    }

    if (!output) {
        throw std::runtime_error(
            "Vocabulary::save: "
            "write failure"
        );
    }
}

// ============================================================
// LOAD
// ============================================================

Vocabulary Vocabulary::load(
    const std::string& path
) {
    std::ifstream input(
        path,
        std::ios::binary
    );

    if (!input) {
        throw std::runtime_error(
            "Vocabulary::load: "
            "cannot open file"
        );
    }

    std::string magic;

    std::getline(
        input,
        magic
    );

    if (magic != "VENLA_VOCAB_V1") {
        throw std::runtime_error(
            "Vocabulary::load: "
            "invalid vocabulary format"
        );
    }

    Vocabulary vocabulary;

    // Clear automatically created vocabulary.
    vocabulary.tokens_.clear();
    vocabulary.token_to_id_.clear();

    auto read_special =
        [&input](
            const std::string& expected_name
        ) -> std::string {

            std::string name;

            std::size_t length;

            input
                >> name
                >> length;

            if (!input ||
                name != expected_name) {

                throw std::runtime_error(
                    "Vocabulary::load: "
                    "invalid special token section"
                );
            }

            char separator;

            input.get(separator);

            std::string token(
                length,
                '\0'
            );

            input.read(
                &token[0],
                static_cast<std::streamsize>(
                    length
                )
            );

            input.get(separator);

            if (!input ||
                separator != '\n') {

                throw std::runtime_error(
                    "Vocabulary::load: "
                    "invalid token data"
                );
            }

            return token;
        };

    const std::string pad =
        read_special("PAD");

    const std::string unk =
        read_special("UNK");

    const std::string bos =
        read_special("BOS");

    const std::string eos =
        read_special("EOS");

    std::string tokens_label;

    std::size_t token_count;

    input
        >> tokens_label
        >> token_count;

    if (!input ||
        tokens_label != "TOKENS") {

        throw std::runtime_error(
            "Vocabulary::load: "
            "missing TOKENS section"
        );
    }

    input.ignore(
        std::numeric_limits<
            std::streamsize
        >::max(),
        '\n'
    );

    for (std::size_t i = 0;
         i < token_count;
         ++i) {

        std::size_t length;

        input >> length;

        char separator;

        input.get(separator);

        if (!input ||
            separator != ' ') {

            throw std::runtime_error(
                "Vocabulary::load: "
                "invalid token length"
            );
        }

        std::string token(
            length,
            '\0'
        );

        input.read(
            &token[0],
            static_cast<std::streamsize>(
                length
            )
        );

        input.get(separator);

        if (!input ||
            separator != '\n') {

            throw std::runtime_error(
                "Vocabulary::load: "
                "invalid token record"
            );
        }

        vocabulary.tokens_.push_back(
            token
        );

        vocabulary.token_to_id_.emplace(
            token,
            i
        );
    }

    vocabulary.set_special_tokens(
        pad,
        unk,
        bos,
        eos
    );

    return vocabulary;
}

} // namespace venla
