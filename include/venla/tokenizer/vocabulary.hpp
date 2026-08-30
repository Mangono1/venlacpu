#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace venla {

// ============================================================
// VOCABULARY
//
// Menyimpan hubungan:
//
//     token <-> id
//
// Vocabulary digunakan oleh tokenizer dan model NLP.
//
// Special tokens:
//
//     <PAD>
//     <UNK>
//     <BOS>
//     <EOS>
//
// ============================================================

class Vocabulary {
public:

    Vocabulary();

    // --------------------------------------------------------
    // TOKEN REGISTRATION
    // --------------------------------------------------------

    std::size_t add_token(
        const std::string& token
    );

    bool contains(
        const std::string& token
    ) const;

    bool contains_id(
        std::size_t id
    ) const;

    // --------------------------------------------------------
    // LOOKUP
    // --------------------------------------------------------

    std::size_t token_to_id(
        const std::string& token
    ) const;

    const std::string& id_to_token(
        std::size_t id
    ) const;

    // --------------------------------------------------------
    // SIZE
    // --------------------------------------------------------

    std::size_t size() const;

    bool empty() const;

    // --------------------------------------------------------
    // SPECIAL TOKENS
    // --------------------------------------------------------

    std::size_t pad_id() const;

    std::size_t unk_id() const;

    std::size_t bos_id() const;

    std::size_t eos_id() const;

    const std::string& pad_token() const;

    const std::string& unk_token() const;

    const std::string& bos_token() const;

    const std::string& eos_token() const;

    // --------------------------------------------------------
    // SPECIAL TOKEN CONFIGURATION
    // --------------------------------------------------------

    void set_special_tokens(
        const std::string& pad,
        const std::string& unk,
        const std::string& bos,
        const std::string& eos
    );

    // --------------------------------------------------------
    // VOCABULARY ACCESS
    // --------------------------------------------------------

    const std::vector<std::string>& tokens() const;

    // --------------------------------------------------------
    // PERSISTENCE
    // --------------------------------------------------------

    void save(
        const std::string& path
    ) const;

    static Vocabulary load(
        const std::string& path
    );

private:

    std::vector<std::string> tokens_;

    std::unordered_map<
        std::string,
        std::size_t
    > token_to_id_;

    std::string pad_token_;

    std::string unk_token_;

    std::string bos_token_;

    std::string eos_token_;

    std::size_t pad_id_;

    std::size_t unk_id_;

    std::size_t bos_id_;

    std::size_t eos_id_;
};

} // namespace venla
