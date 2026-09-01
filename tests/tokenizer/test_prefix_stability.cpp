#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "venla/tokenizer/tokenizer.hpp"

static void check_prefix(
    venla::BPETokenizer& tokenizer,
    const std::string& prefix,
    const std::string& continuation
) {
    const std::vector<std::size_t> prefix_ids =
        tokenizer.encode(prefix);

    const std::vector<std::size_t> full_ids =
        tokenizer.encode(
            prefix + continuation
        );

    assert(
        full_ids.size() >=
        prefix_ids.size()
    );

    for (
        std::size_t i = 0;
        i < prefix_ids.size();
        ++i
    ) {
        if (prefix_ids[i] != full_ids[i]) {
            std::cerr
                << "PREFIX STABILITY FAILURE\n"
                << "prefix      = "
                << prefix
                << "\ncontinuation = "
                << continuation
                << "\nposition    = "
                << i
                << "\nprefix id   = "
                << prefix_ids[i]
                << "\nfull id     = "
                << full_ids[i]
                << "\n";

            assert(false);
        }
    }
}

int main() {

    const std::string text =
        "Di sebuah desa, hiduplah Bima. "
        "Bima adalah anak yang jujur. "
        "Bima menjaga keris pusaka Nusantara. "
        "Keris itu sangat sakti dan bersinar.";

    venla::BPETokenizer tokenizer;

    tokenizer.train(text);

    check_prefix(
        tokenizer,
        "Bima",
        " menjaga"
    );

    check_prefix(
        tokenizer,
        "Bima menjaga",
        " keris"
    );

    check_prefix(
        tokenizer,
        "Bima menjaga keris",
        " pusaka"
    );

    check_prefix(
        tokenizer,
        "Keris",
        " itu"
    );

    check_prefix(
        tokenizer,
        "Keris itu",
        " sangat"
    );

    check_prefix(
        tokenizer,
        "Keris itu sangat",
        " sakti"
    );

    check_prefix(
        tokenizer,
        "Di sebuah desa, hiduplah",
        " Bima"
    );

    // Round-trip must remain exact.
    const std::vector<std::size_t> ids =
        tokenizer.encode(text);

    assert(
        tokenizer.decode(ids) == text
    );

    std::cout
        << "VENLACPU tokenizer prefix stability tests passed\n";

    return 0;
}
