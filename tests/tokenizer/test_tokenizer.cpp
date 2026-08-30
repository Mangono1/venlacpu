#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "venla/tokenizer/tokenizer.hpp"
#include "venla/tokenizer/vocabulary.hpp"

int main() {

    // ========================================================
    // BASIC VOCABULARY
    // ========================================================

    {
        venla::Vocabulary vocabulary;

        assert(
            vocabulary.size() == 4
        );

        assert(
            vocabulary.contains("<PAD>")
        );

        assert(
            vocabulary.contains("<UNK>")
        );

        assert(
            vocabulary.contains("<BOS>")
        );

        assert(
            vocabulary.contains("<EOS>")
        );

        const std::size_t hello_id =
            vocabulary.add_token(
                "hello"
            );

        assert(
            vocabulary.contains("hello")
        );

        assert(
            vocabulary.token_to_id("hello")
            == hello_id
        );

        assert(
            vocabulary.id_to_token(hello_id)
            == "hello"
        );

        assert(
            vocabulary.token_to_id("unknown")
            == vocabulary.unk_id()
        );
    }

    // ========================================================
    // BPE TRAINING
    // ========================================================

    venla::BPETokenizer::Config config;

    config.vocab_size = 32;

    config.min_frequency = 2;

    config.add_bos = false;

    config.add_eos = false;

    venla::BPETokenizer tokenizer(
        config
    );

    tokenizer.train(
        std::vector<std::string>{
            "hello hello hello",
            "hello hello",
            "hello world",
            "hello world",
            "hello world"
        }
    );

    assert(
        tokenizer.trained()
    );

    assert(
        tokenizer.vocabulary().size()
        <= 32
    );

    assert(
        tokenizer.merge_count() > 0
    );

    // ========================================================
    // ENCODE
    // ========================================================

    const std::string text =
        "hello world";

    std::vector<std::size_t> ids =
        tokenizer.encode(text);

    assert(
        !ids.empty()
    );

    for (const std::size_t id : ids) {
        assert(
            tokenizer.vocabulary()
                .contains_id(id)
        );
    }

    // ========================================================
    // DECODE
    // ========================================================

    const std::string decoded =
        tokenizer.decode(ids);

    assert(
        decoded == text
    );

    // ========================================================
    // UTF-8
    // ========================================================

    const std::string indonesian =
        "saya belajar bahasa Indonesia";

    const std::vector<std::size_t>
        indonesian_ids =
            tokenizer.encode(
                indonesian
            );

    const std::string
        indonesian_decoded =
            tokenizer.decode(
                indonesian_ids
            );

    assert(
        indonesian_decoded ==
        indonesian
    );

    // ========================================================
    // SPECIAL TOKENS
    // ========================================================

    std::vector<std::size_t>
        special_ids =
            tokenizer.encode(
                "hello",
                true,
                true
            );

    assert(
        special_ids.size() >= 3
    );

    assert(
        special_ids.front()
        ==
        tokenizer.vocabulary().bos_id()
    );

    assert(
        special_ids.back()
        ==
        tokenizer.vocabulary().eos_id()
    );

    const std::string
        special_decoded =
            tokenizer.decode(
                special_ids
            );

    assert(
        special_decoded == "hello"
    );

    // ========================================================
    // SAVE / LOAD
    // ========================================================

    const std::string path =
        "/data/data/com.termux/files/home/venlacpu/build/test_tokenizer.v1";

    tokenizer.save(
        path
    );

    venla::BPETokenizer loaded =
        venla::BPETokenizer::load(
            path
        );

    assert(
        loaded.trained()
    );

    assert(
        loaded.vocabulary().size()
        ==
        tokenizer.vocabulary().size()
    );

    assert(
        loaded.merge_count()
        ==
        tokenizer.merge_count()
    );

    const std::vector<std::size_t>
        loaded_ids =
            loaded.encode(text);

    assert(
        loaded_ids == ids
    );

    assert(
        loaded.decode(loaded_ids)
        == text
    );

    std::remove(
        path.c_str()
    );

    std::remove(
        (path + ".vocab").c_str()
    );

    std::cout
        << "VENLACPU tokenizer tests passed\n";

    return 0;
}
