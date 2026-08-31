#include "venla/nn/multi_head_attention.hpp"

#include "venla/autograd/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace venla {

namespace {

// ============================================================
// VALIDATION
// ============================================================

void require_float32(
    const Tensor& tensor,
    const char* operation
) {
    if (tensor.dtype() != DType::Float32) {

        std::ostringstream message;

        message
            << operation
            << ": only Float32 is currently supported, got "
            << dtype_name(tensor.dtype());

        throw std::runtime_error(
            message.str()
        );
    }

    if (!tensor.device().is_cpu()) {

        throw std::runtime_error(
            std::string(operation) +
            ": only CPU device is currently supported"
        );
    }
}

// ============================================================
// MATRIX PROJECTION
//
// input:
//   [rows, embed]
//
// weight:
//   [embed, embed]
//
// bias:
//   [embed]
//
// output:
//   [rows, embed]
// ============================================================

Tensor project_2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    bool use_bias
) {
    const std::size_t rows =
        input.shape()[0];

    const std::size_t embed =
        input.shape()[1];

    Tensor output =
        Tensor::zeros(
            {rows, embed},
            DType::Float32,
            input.device()
        );

    const float* x =
        input.data_as<float>();

    const float* w =
        weight.data_as<float>();

    const float* b =
        use_bias
            ? bias.data_as<float>()
            : nullptr;

    float* y =
        output.data_as<float>();

    for (std::size_t row = 0;
         row < rows;
         ++row) {

        for (std::size_t out = 0;
             out < embed;
             ++out) {

            double value = 0.0;

            for (std::size_t in = 0;
                 in < embed;
                 ++in) {

                value +=
                    static_cast<double>(
                        x[
                            row * embed +
                            in
                        ]
                    ) *
                    static_cast<double>(
                        w[
                            in * embed +
                            out
                        ]
                    );
            }

            if (b != nullptr) {
                value +=
                    static_cast<double>(
                        b[out]
                    );
            }

            y[
                row * embed +
                out
            ] =
                static_cast<float>(
                    value
                );
        }
    }

    return output;
}

// ============================================================
// NORMAL ATTENTION
//
// q/k/v:
//   [seq, embed]
//
// output:
//   [seq, embed]
// ============================================================

Tensor attention_2d(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::size_t num_heads,
    bool causal
) {
    const std::size_t query_length =
        q.shape()[0];

    const std::size_t key_length =
        k.shape()[0];

    const std::size_t embed_dim =
        q.shape()[1];

    const std::size_t head_dim =
        embed_dim /
        num_heads;

    Tensor output =
        Tensor::zeros(
            {query_length, embed_dim},
            DType::Float32,
            q.device()
        );

    const float* q_data =
        q.data_as<float>();

    const float* k_data =
        k.data_as<float>();

    const float* v_data =
        v.data_as<float>();

    float* output_data =
        output.data_as<float>();

    const double scale =
        1.0 /
        std::sqrt(
            static_cast<double>(
                head_dim
            )
        );

    for (std::size_t head = 0;
         head < num_heads;
         ++head) {

        const std::size_t head_offset =
            head * head_dim;

        for (std::size_t query = 0;
             query < query_length;
             ++query) {

            std::vector<double> scores(
                key_length,
                -std::numeric_limits<double>::infinity()
            );

            double maximum =
                -std::numeric_limits<double>::infinity();

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (causal &&
                    key_index > query) {

                    continue;
                }

                double score = 0.0;

                for (std::size_t d = 0;
                     d < head_dim;
                     ++d) {

                    score +=
                        static_cast<double>(
                            q_data[
                                query * embed_dim +
                                head_offset +
                                d
                            ]
                        ) *
                        static_cast<double>(
                            k_data[
                                key_index * embed_dim +
                                head_offset +
                                d
                            ]
                        );
                }

                score *= scale;

                scores[key_index] =
                    score;

                maximum =
                    std::max(
                        maximum,
                        score
                    );
            }

            double denominator = 0.0;

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (!std::isfinite(
                        scores[key_index]
                    )) {

                    continue;
                }

                scores[key_index] =
                    std::exp(
                        scores[key_index] -
                        maximum
                    );

                denominator +=
                    scores[key_index];
            }

            if (denominator == 0.0) {
                continue;
            }

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (!std::isfinite(
                        scores[key_index]
                    )) {

                    continue;
                }

                const double probability =
                    scores[key_index] /
                    denominator;

                for (std::size_t d = 0;
                     d < head_dim;
                     ++d) {

                    output_data[
                        query * embed_dim +
                        head_offset +
                        d
                    ] +=
                        static_cast<float>(
                            probability *
                            static_cast<double>(
                                v_data[
                                    key_index * embed_dim +
                                    head_offset +
                                    d
                                ]
                            )
                        );
                }
            }
        }
    }

    return output;
}

// ============================================================
// CACHED ATTENTION
//
// q:
//   [new_seq, embed]
//
// k/v:
//   [total_seq, embed]
//
// old_sequence:
//   number of tokens that existed before this call
//
// For autoregressive decoding:
//
//   query 0 -> key positions <= old_seq
//   query 1 -> key positions <= old_seq + 1
//   ...
//
// This allows multi-token cached appends while preserving
// causal ordering.
// ============================================================

Tensor cached_attention_2d(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::size_t num_heads,
    bool causal,
    std::size_t old_sequence
) {
    const std::size_t query_length =
        q.shape()[0];

    const std::size_t key_length =
        k.shape()[0];

    const std::size_t embed_dim =
        q.shape()[1];

    const std::size_t head_dim =
        embed_dim /
        num_heads;

    Tensor output =
        Tensor::zeros(
            {query_length, embed_dim},
            DType::Float32,
            q.device()
        );

    const float* q_data =
        q.data_as<float>();

    const float* k_data =
        k.data_as<float>();

    const float* v_data =
        v.data_as<float>();

    float* output_data =
        output.data_as<float>();

    const double scale =
        1.0 /
        std::sqrt(
            static_cast<double>(
                head_dim
            )
        );

    for (std::size_t head = 0;
         head < num_heads;
         ++head) {

        const std::size_t head_offset =
            head * head_dim;

        for (std::size_t query = 0;
             query < query_length;
             ++query) {

            const std::size_t absolute_query =
                old_sequence +
                query;

            std::vector<double> scores(
                key_length,
                -std::numeric_limits<double>::infinity()
            );

            double maximum =
                -std::numeric_limits<double>::infinity();

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (causal &&
                    key_index > absolute_query) {

                    continue;
                }

                double score = 0.0;

                for (std::size_t d = 0;
                     d < head_dim;
                     ++d) {

                    score +=
                        static_cast<double>(
                            q_data[
                                query * embed_dim +
                                head_offset +
                                d
                            ]
                        ) *
                        static_cast<double>(
                            k_data[
                                key_index * embed_dim +
                                head_offset +
                                d
                            ]
                        );
                }

                score *= scale;

                scores[key_index] =
                    score;

                maximum =
                    std::max(
                        maximum,
                        score
                    );
            }

            double denominator = 0.0;

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (!std::isfinite(
                        scores[key_index]
                    )) {

                    continue;
                }

                scores[key_index] =
                    std::exp(
                        scores[key_index] -
                        maximum
                    );

                denominator +=
                    scores[key_index];
            }

            if (denominator == 0.0) {
                continue;
            }

            for (std::size_t key_index = 0;
                 key_index < key_length;
                 ++key_index) {

                if (!std::isfinite(
                        scores[key_index]
                    )) {

                    continue;
                }

                const double probability =
                    scores[key_index] /
                    denominator;

                for (std::size_t d = 0;
                     d < head_dim;
                     ++d) {

                    output_data[
                        query * embed_dim +
                        head_offset +
                        d
                    ] +=
                        static_cast<float>(
                            probability *
                            static_cast<double>(
                                v_data[
                                    key_index * embed_dim +
                                    head_offset +
                                    d
                                ]
                            )
                        );
                }
            }
        }
    }

    return output;
}

// ============================================================
// AUTOGRAD NODE
// ============================================================

std::shared_ptr<AutogradNode>
make_mha_node(
    const Tensor& input,
    const Tensor& q_weight,
    const Tensor& k_weight,
    const Tensor& v_weight,
    const Tensor& out_weight,
    const Tensor& q_bias,
    const Tensor& k_bias,
    const Tensor& v_bias,
    const Tensor& out_bias,
    std::size_t num_heads,
    bool use_bias,
    bool causal
) {
    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{
            input,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            q_bias,
            k_bias,
            v_bias,
            out_bias
        },

        [
            input,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            q_bias,
            k_bias,
            v_bias,
            out_bias,
            num_heads,
            use_bias,
            causal
        ](const Tensor& gradient) {

            const std::size_t embed_dim =
                input.shape()[
                    input.ndim() - 1
                ];

            const std::size_t head_dim =
                embed_dim /
                num_heads;

            const std::size_t batch =
                input.ndim() == 3
                    ? input.shape()[0]
                    : 1;

            const std::size_t sequence_length =
                input.shape()[
                    input.ndim() - 2
                ];

            const float* x =
                input.data_as<float>();

            const float* dy =
                gradient.data_as<float>();

            const float* wq =
                q_weight.data_as<float>();

            const float* wk =
                k_weight.data_as<float>();

            const float* wv =
                v_weight.data_as<float>();

            const float* wo =
                out_weight.data_as<float>();

            const float* bq =
                use_bias
                    ? q_bias.data_as<float>()
                    : nullptr;

            const float* bk =
                use_bias
                    ? k_bias.data_as<float>()
                    : nullptr;

            const float* bv =
                use_bias
                    ? v_bias.data_as<float>()
                    : nullptr;

            Tensor dx;

            if (input.requires_grad()) {

                dx =
                    Tensor::zeros(
                        input.shape(),
                        DType::Float32,
                        input.device()
                    );
            }

            Tensor dwq;

            if (q_weight.requires_grad()) {

                dwq =
                    Tensor::zeros(
                        q_weight.shape(),
                        DType::Float32,
                        q_weight.device()
                    );
            }

            Tensor dwk;

            if (k_weight.requires_grad()) {

                dwk =
                    Tensor::zeros(
                        k_weight.shape(),
                        DType::Float32,
                        k_weight.device()
                    );
            }

            Tensor dwv;

            if (v_weight.requires_grad()) {

                dwv =
                    Tensor::zeros(
                        v_weight.shape(),
                        DType::Float32,
                        v_weight.device()
                    );
            }

            Tensor dwo;

            if (out_weight.requires_grad()) {

                dwo =
                    Tensor::zeros(
                        out_weight.shape(),
                        DType::Float32,
                        out_weight.device()
                    );
            }

            Tensor dbq;

            if (use_bias &&
                q_bias.requires_grad()) {

                dbq =
                    Tensor::zeros(
                        q_bias.shape(),
                        DType::Float32,
                        q_bias.device()
                    );
            }

            Tensor dbk;

            if (use_bias &&
                k_bias.requires_grad()) {

                dbk =
                    Tensor::zeros(
                        k_bias.shape(),
                        DType::Float32,
                        k_bias.device()
                    );
            }

            Tensor dbv;

            if (use_bias &&
                v_bias.requires_grad()) {

                dbv =
                    Tensor::zeros(
                        v_bias.shape(),
                        DType::Float32,
                        v_bias.device()
                    );
            }

            Tensor dbo;

            if (use_bias &&
                out_bias.requires_grad()) {

                dbo =
                    Tensor::zeros(
                        out_bias.shape(),
                        DType::Float32,
                        out_bias.device()
                    );
            }

            float* dx_data =
                input.requires_grad()
                    ? dx.data_as<float>()
                    : nullptr;

            float* dwq_data =
                q_weight.requires_grad()
                    ? dwq.data_as<float>()
                    : nullptr;

            float* dwk_data =
                k_weight.requires_grad()
                    ? dwk.data_as<float>()
                    : nullptr;

            float* dwv_data =
                v_weight.requires_grad()
                    ? dwv.data_as<float>()
                    : nullptr;

            float* dwo_data =
                out_weight.requires_grad()
                    ? dwo.data_as<float>()
                    : nullptr;

            float* dbq_data =
                use_bias &&
                q_bias.requires_grad()
                    ? dbq.data_as<float>()
                    : nullptr;

            float* dbk_data =
                use_bias &&
                k_bias.requires_grad()
                    ? dbk.data_as<float>()
                    : nullptr;

            float* dbv_data =
                use_bias &&
                v_bias.requires_grad()
                    ? dbv.data_as<float>()
                    : nullptr;

            float* dbo_data =
                use_bias &&
                out_bias.requires_grad()
                    ? dbo.data_as<float>()
                    : nullptr;

            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {

                const std::size_t base =
                    batch_index *
                    sequence_length *
                    embed_dim;

                std::vector<float> q(
                    sequence_length * embed_dim
                );

                std::vector<float> k(
                    sequence_length * embed_dim
                );

                std::vector<float> v(
                    sequence_length * embed_dim
                );

                for (std::size_t row = 0;
                     row < sequence_length;
                     ++row) {

                    for (std::size_t out = 0;
                         out < embed_dim;
                         ++out) {

                        double q_value = 0.0;
                        double k_value = 0.0;
                        double v_value = 0.0;

                        for (std::size_t in = 0;
                             in < embed_dim;
                             ++in) {

                            const double value =
                                static_cast<double>(
                                    x[
                                        base +
                                        row * embed_dim +
                                        in
                                    ]
                                );

                            q_value +=
                                value *
                                static_cast<double>(
                                    wq[
                                        in * embed_dim +
                                        out
                                    ]
                                );

                            k_value +=
                                value *
                                static_cast<double>(
                                    wk[
                                        in * embed_dim +
                                        out
                                    ]
                                );

                            v_value +=
                                value *
                                static_cast<double>(
                                    wv[
                                        in * embed_dim +
                                        out
                                    ]
                                );
                        }

                        if (use_bias) {

                            q_value += bq[out];
                            k_value += bk[out];
                            v_value += bv[out];
                        }

                        q[
                            row * embed_dim +
                            out
                        ] =
                            static_cast<float>(
                                q_value
                            );

                        k[
                            row * embed_dim +
                            out
                        ] =
                            static_cast<float>(
                                k_value
                            );

                        v[
                            row * embed_dim +
                            out
                        ] =
                            static_cast<float>(
                                v_value
                            );
                    }
                }

                std::vector<float> context(
                    sequence_length * embed_dim,
                    0.0f
                );

                const double scale =
                    1.0 /
                    std::sqrt(
                        static_cast<double>(
                            head_dim
                        )
                    );

                std::vector<float> attention(
                    num_heads *
                    sequence_length *
                    sequence_length,
                    0.0f
                );

                for (std::size_t head = 0;
                     head < num_heads;
                     ++head) {

                    const std::size_t head_offset =
                        head * head_dim;

                    for (std::size_t query = 0;
                         query < sequence_length;
                         ++query) {

                        std::vector<double> scores(
                            sequence_length,
                            -std::numeric_limits<double>::infinity()
                        );

                        double maximum =
                            -std::numeric_limits<double>::infinity();

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            if (causal &&
                                key_index > query) {

                                continue;
                            }

                            double score = 0.0;

                            for (std::size_t d = 0;
                                 d < head_dim;
                                 ++d) {

                                score +=
                                    static_cast<double>(
                                        q[
                                            query * embed_dim +
                                            head_offset +
                                            d
                                        ]
                                    ) *
                                    static_cast<double>(
                                        k[
                                            key_index * embed_dim +
                                            head_offset +
                                            d
                                        ]
                                    );
                            }

                            score *= scale;

                            scores[key_index] =
                                score;

                            maximum =
                                std::max(
                                    maximum,
                                    score
                                );
                        }

                        double denominator = 0.0;

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            if (!std::isfinite(
                                    scores[key_index]
                                )) {

                                continue;
                            }

                            scores[key_index] =
                                std::exp(
                                    scores[key_index] -
                                    maximum
                                );

                            denominator +=
                                scores[key_index];
                        }

                        if (denominator == 0.0) {
                            continue;
                        }

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            if (!std::isfinite(
                                    scores[key_index]
                                )) {

                                continue;
                            }

                            attention[
                                (
                                    head *
                                    sequence_length +
                                    query
                                ) *
                                sequence_length +
                                key_index
                            ] =
                                static_cast<float>(
                                    scores[key_index] /
                                    denominator
                                );
                        }
                    }
                }

                for (std::size_t head = 0;
                     head < num_heads;
                     ++head) {

                    const std::size_t head_offset =
                        head * head_dim;

                    for (std::size_t query = 0;
                         query < sequence_length;
                         ++query) {

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            const float probability =
                                attention[
                                    (
                                        head *
                                        sequence_length +
                                        query
                                    ) *
                                    sequence_length +
                                    key_index
                                ];

                            if (probability == 0.0f) {
                                continue;
                            }

                            for (std::size_t d = 0;
                                 d < head_dim;
                                 ++d) {

                                context[
                                    query * embed_dim +
                                    head_offset +
                                    d
                                ] +=
                                    probability *
                                    v[
                                        key_index * embed_dim +
                                        head_offset +
                                        d
                                    ];
                            }
                        }
                    }
                }

                std::vector<float> dcontext(
                    sequence_length * embed_dim,
                    0.0f
                );

                for (std::size_t row = 0;
                     row < sequence_length;
                     ++row) {

                    for (std::size_t out = 0;
                         out < embed_dim;
                         ++out) {

                        const float upstream =
                            dy[
                                base +
                                row * embed_dim +
                                out
                            ];

                        if (use_bias &&
                            dbo_data != nullptr) {

                            dbo_data[out] +=
                                upstream;
                        }

                        for (std::size_t in = 0;
                             in < embed_dim;
                             ++in) {

                            if (dwo_data != nullptr) {

                                dwo_data[
                                    in * embed_dim +
                                    out
                                ] +=
                                    context[
                                        row * embed_dim +
                                        in
                                    ] *
                                    upstream;
                            }

                            dcontext[
                                row * embed_dim +
                                in
                            ] +=
                                wo[
                                    in * embed_dim +
                                    out
                                ] *
                                upstream;
                        }
                    }
                }

                std::vector<float> dq(
                    sequence_length * embed_dim,
                    0.0f
                );

                std::vector<float> dk(
                    sequence_length * embed_dim,
                    0.0f
                );

                std::vector<float> dv(
                    sequence_length * embed_dim,
                    0.0f
                );

                for (std::size_t head = 0;
                     head < num_heads;
                     ++head) {

                    const std::size_t head_offset =
                        head * head_dim;

                    for (std::size_t query = 0;
                         query < sequence_length;
                         ++query) {

                        std::vector<double> d_attention(
                            sequence_length,
                            0.0
                        );

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            const float probability =
                                attention[
                                    (
                                        head *
                                        sequence_length +
                                        query
                                    ) *
                                    sequence_length +
                                    key_index
                                ];

                            if (probability == 0.0f) {
                                continue;
                            }

                            double value = 0.0;

                            for (std::size_t d = 0;
                                 d < head_dim;
                                 ++d) {

                                value +=
                                    static_cast<double>(
                                        dcontext[
                                            query * embed_dim +
                                            head_offset +
                                            d
                                        ]
                                    ) *
                                    static_cast<double>(
                                        v[
                                            key_index * embed_dim +
                                            head_offset +
                                            d
                                        ]
                                    );

                                dv[
                                    key_index * embed_dim +
                                    head_offset +
                                    d
                                ] +=
                                    probability *
                                    dcontext[
                                        query * embed_dim +
                                        head_offset +
                                        d
                                    ];
                            }

                            d_attention[key_index] =
                                value;
                        }

                        double weighted_sum = 0.0;

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            const float probability =
                                attention[
                                    (
                                        head *
                                        sequence_length +
                                        query
                                    ) *
                                    sequence_length +
                                    key_index
                                ];

                            weighted_sum +=
                                d_attention[key_index] *
                                static_cast<double>(
                                    probability
                                );
                        }

                        for (std::size_t key_index = 0;
                             key_index < sequence_length;
                             ++key_index) {

                            const float probability =
                                attention[
                                    (
                                        head *
                                        sequence_length +
                                        query
                                    ) *
                                    sequence_length +
                                    key_index
                                ];

                            if (probability == 0.0f) {
                                continue;
                            }

                            const double d_score =
                                static_cast<double>(
                                    probability
                                ) *
                                (
                                    d_attention[key_index] -
                                    weighted_sum
                                ) *
                                scale;

                            for (std::size_t d = 0;
                                 d < head_dim;
                                 ++d) {

                                dq[
                                    query * embed_dim +
                                    head_offset +
                                    d
                                ] +=
                                    static_cast<float>(
                                        d_score *
                                        static_cast<double>(
                                            k[
                                                key_index * embed_dim +
                                                head_offset +
                                                d
                                            ]
                                        )
                                    );

                                dk[
                                    key_index * embed_dim +
                                    head_offset +
                                    d
                                ] +=
                                    static_cast<float>(
                                        d_score *
                                        static_cast<double>(
                                            q[
                                                query * embed_dim +
                                                head_offset +
                                                d
                                            ]
                                        )
                                    );
                            }
                        }
                    }
                }

                for (std::size_t row = 0;
                     row < sequence_length;
                     ++row) {

                    for (std::size_t out = 0;
                         out < embed_dim;
                         ++out) {

                        const float dq_value =
                            dq[
                                row * embed_dim +
                                out
                            ];

                        const float dk_value =
                            dk[
                                row * embed_dim +
                                out
                            ];

                        const float dv_value =
                            dv[
                                row * embed_dim +
                                out
                            ];

                        if (dbq_data != nullptr) {
                            dbq_data[out] +=
                                dq_value;
                        }

                        if (dbk_data != nullptr) {
                            dbk_data[out] +=
                                dk_value;
                        }

                        if (dbv_data != nullptr) {
                            dbv_data[out] +=
                                dv_value;
                        }

                        for (std::size_t in = 0;
                             in < embed_dim;
                             ++in) {

                            const float x_value =
                                x[
                                    base +
                                    row * embed_dim +
                                    in
                                ];

                            if (dwq_data != nullptr) {

                                dwq_data[
                                    in * embed_dim +
                                    out
                                ] +=
                                    x_value *
                                    dq_value;
                            }

                            if (dwk_data != nullptr) {

                                dwk_data[
                                    in * embed_dim +
                                    out
                                ] +=
                                    x_value *
                                    dk_value;
                            }

                            if (dwv_data != nullptr) {

                                dwv_data[
                                    in * embed_dim +
                                    out
                                ] +=
                                    x_value *
                                    dv_value;
                            }

                            if (dx_data != nullptr) {

                                dx_data[
                                    base +
                                    row * embed_dim +
                                    in
                                ] +=
                                    wq[
                                        in * embed_dim +
                                        out
                                    ] *
                                    dq_value;

                                dx_data[
                                    base +
                                    row * embed_dim +
                                    in
                                ] +=
                                    wk[
                                        in * embed_dim +
                                        out
                                    ] *
                                    dk_value;

                                dx_data[
                                    base +
                                    row * embed_dim +
                                    in
                                ] +=
                                    wv[
                                        in * embed_dim +
                                        out
                                    ] *
                                    dv_value;
                            }
                        }
                    }
                }
            }

            if (input.requires_grad()) {

                input.accumulate_grad(dx);

                if (input.grad_state()->grad_fn) {

                    input.grad_state()
                        ->grad_fn
                        ->backward(dx);
                }
            }

            if (q_weight.requires_grad()) {

                q_weight.accumulate_grad(dwq);

                if (q_weight.grad_state()->grad_fn) {

                    q_weight.grad_state()
                        ->grad_fn
                        ->backward(dwq);
                }
            }

            if (k_weight.requires_grad()) {

                k_weight.accumulate_grad(dwk);

                if (k_weight.grad_state()->grad_fn) {

                    k_weight.grad_state()
                        ->grad_fn
                        ->backward(dwk);
                }
            }

            if (v_weight.requires_grad()) {

                v_weight.accumulate_grad(dwv);

                if (v_weight.grad_state()->grad_fn) {

                    v_weight.grad_state()
                        ->grad_fn
                        ->backward(dwv);
                }
            }

            if (out_weight.requires_grad()) {

                out_weight.accumulate_grad(dwo);

                if (out_weight.grad_state()->grad_fn) {

                    out_weight.grad_state()
                        ->grad_fn
                        ->backward(dwo);
                }
            }

            if (use_bias &&
                q_bias.requires_grad()) {

                q_bias.accumulate_grad(dbq);
            }

            if (use_bias &&
                k_bias.requires_grad()) {

                k_bias.accumulate_grad(dbk);
            }

            if (use_bias &&
                v_bias.requires_grad()) {

                v_bias.accumulate_grad(dbv);
            }

            if (use_bias &&
                out_bias.requires_grad()) {

                out_bias.accumulate_grad(dbo);
            }
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

MultiHeadAttention::MultiHeadAttention(
    std::size_t embed_dim,
    std::size_t num_heads,
    bool use_bias,
    bool causal
)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(0),
      use_bias_(use_bias),
      causal_(causal),
      q_weight_(),
      k_weight_(),
      v_weight_(),
      out_weight_(),
      q_bias_(),
      k_bias_(),
      v_bias_(),
      out_bias_() {

    if (embed_dim == 0) {

        throw std::invalid_argument(
            "MultiHeadAttention: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {

        throw std::invalid_argument(
            "MultiHeadAttention: "
            "num_heads must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {

        throw std::invalid_argument(
            "MultiHeadAttention: "
            "embed_dim must be divisible by num_heads"
        );
    }

    head_dim_ =
        embed_dim /
        num_heads;

    q_weight_ =
        Tensor::empty(
            {embed_dim_, embed_dim_},
            DType::Float32,
            Device::cpu()
        );

    k_weight_ =
        Tensor::empty(
            {embed_dim_, embed_dim_},
            DType::Float32,
            Device::cpu()
        );

    v_weight_ =
        Tensor::empty(
            {embed_dim_, embed_dim_},
            DType::Float32,
            Device::cpu()
        );

    out_weight_ =
        Tensor::empty(
            {embed_dim_, embed_dim_},
            DType::Float32,
            Device::cpu()
        );

    if (use_bias_) {

        q_bias_ =
            Tensor::zeros(
                {embed_dim_},
                DType::Float32,
                Device::cpu()
            );

        k_bias_ =
            Tensor::zeros(
                {embed_dim_},
                DType::Float32,
                Device::cpu()
            );

        v_bias_ =
            Tensor::zeros(
                {embed_dim_},
                DType::Float32,
                Device::cpu()
            );

        out_bias_ =
            Tensor::zeros(
                {embed_dim_},
                DType::Float32,
                Device::cpu()
            );
    }

    q_weight_.requires_grad_(true);
    k_weight_.requires_grad_(true);
    v_weight_.requires_grad_(true);
    out_weight_.requires_grad_(true);

    if (use_bias_) {

        q_bias_.requires_grad_(true);
        k_bias_.requires_grad_(true);
        v_bias_.requires_grad_(true);
        out_bias_.requires_grad_(true);
    }

    reset_parameters();
}

// ============================================================
// RESET PARAMETERS
// ============================================================

void MultiHeadAttention::reset_parameters() {

    const float limit =
        std::sqrt(
            6.0f /
            static_cast<float>(
                embed_dim_ +
                embed_dim_
            )
        );

    std::mt19937 generator(
        0x4D484153u
    );

    std::uniform_real_distribution<float>
        distribution(
            -limit,
            limit
        );

    Tensor* weights[] = {
        &q_weight_,
        &k_weight_,
        &v_weight_,
        &out_weight_
    };

    for (Tensor* tensor : weights) {

        float* data =
            tensor->data_as<float>();

        for (std::size_t i = 0;
             i < tensor->numel();
             ++i) {

            data[i] =
                distribution(
                    generator
                );
        }
    }

    if (use_bias_) {

        Tensor* biases[] = {
            &q_bias_,
            &k_bias_,
            &v_bias_,
            &out_bias_
        };

        for (Tensor* tensor : biases) {

            float* data =
                tensor->data_as<float>();

            for (std::size_t i = 0;
                 i < tensor->numel();
                 ++i) {

                data[i] = 0.0f;
            }
        }
    }
}

// ============================================================
// NORMAL FORWARD
// ============================================================

Tensor MultiHeadAttention::forward(
    const Tensor& input
) const {

    require_float32(
        input,
        "MultiHeadAttention::forward"
    );

    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "MultiHeadAttention::forward: "
            "input must be 2D or 3D"
        );
    }

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim_) {

        std::ostringstream message;

        message
            << "MultiHeadAttention::forward: "
            << "input last dimension must be "
            << embed_dim_
            << ", got "
            << features;

        throw std::runtime_error(
            message.str()
        );
    }

    const std::size_t sequence_length =
        input.shape()[
            input.ndim() - 2
        ];

    if (sequence_length == 0) {

        throw std::runtime_error(
            "MultiHeadAttention::forward: "
            "sequence length must be greater than zero"
        );
    }

    const std::size_t batch =
        input.ndim() == 3
            ? input.shape()[0]
            : 1;

    Tensor result =
        Tensor::zeros(
            input.shape(),
            DType::Float32,
            input.device()
        );

    const float* input_data =
        input.data_as<float>();

    float* result_data =
        result.data_as<float>();

    const std::size_t sequence_elements =
        sequence_length *
        embed_dim_;

    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {

        Tensor sequence =
            Tensor::zeros(
                {sequence_length, embed_dim_},
                DType::Float32,
                input.device()
            );

        float* sequence_data =
            sequence.data_as<float>();

        for (std::size_t i = 0;
             i < sequence_elements;
             ++i) {

            sequence_data[i] =
                input_data[
                    batch_index *
                    sequence_elements +
                    i
                ];
        }

        Tensor q =
            project_2d(
                sequence,
                q_weight_,
                q_bias_,
                use_bias_
            );

        Tensor k =
            project_2d(
                sequence,
                k_weight_,
                k_bias_,
                use_bias_
            );

        Tensor v =
            project_2d(
                sequence,
                v_weight_,
                v_bias_,
                use_bias_
            );

        Tensor context =
            attention_2d(
                q,
                k,
                v,
                num_heads_,
                causal_
            );

        Tensor projected =
            project_2d(
                context,
                out_weight_,
                out_bias_,
                use_bias_
            );

        const float* projected_data =
            projected.data_as<float>();

        for (std::size_t i = 0;
             i < sequence_elements;
             ++i) {

            result_data[
                batch_index *
                sequence_elements +
                i
            ] =
                projected_data[i];
        }
    }

    if (input.requires_grad() ||
        q_weight_.requires_grad() ||
        k_weight_.requires_grad() ||
        v_weight_.requires_grad() ||
        out_weight_.requires_grad() ||
        (use_bias_ &&
         (
             q_bias_.requires_grad() ||
             k_bias_.requires_grad() ||
             v_bias_.requires_grad() ||
             out_bias_.requires_grad()
         ))) {

        result.set_grad_fn(
            make_mha_node(
                input,
                q_weight_,
                k_weight_,
                v_weight_,
                out_weight_,
                q_bias_,
                k_bias_,
                v_bias_,
                out_bias_,
                num_heads_,
                use_bias_,
                causal_
            )
        );
    }

    return result;
}

// ============================================================
// CACHED FORWARD
// ============================================================

Tensor MultiHeadAttention::forward_cached(
    const Tensor& input,
    KVCache& cache
) const {

    require_float32(
        input,
        "MultiHeadAttention::forward_cached"
    );

    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "MultiHeadAttention::forward_cached: "
            "input must be 2D or 3D"
        );
    }

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim_) {

        std::ostringstream message;

        message
            << "MultiHeadAttention::forward_cached: "
            << "input last dimension must be "
            << embed_dim_
            << ", got "
            << features;

        throw std::runtime_error(
            message.str()
        );
    }

    const std::size_t new_sequence =
        input.shape()[
            input.ndim() - 2
        ];

    if (new_sequence == 0) {

        throw std::runtime_error(
            "MultiHeadAttention::forward_cached: "
            "sequence length must be greater than zero"
        );
    }

    const std::size_t batch =
        input.ndim() == 3
            ? input.shape()[0]
            : 1;

    // --------------------------------------------------------
    // Cache rank must match input rank.
    // --------------------------------------------------------

    if (cache.initialized() &&
        (
            (input.ndim() == 2 &&
             cache.key().ndim() != 2) ||

            (input.ndim() == 3 &&
             cache.key().ndim() != 3)
        )) {

        throw std::runtime_error(
            "MultiHeadAttention::forward_cached: "
            "input rank does not match KV cache rank"
        );
    }

    // --------------------------------------------------------
    // Batch validation.
    // --------------------------------------------------------

    if (cache.initialized() &&
        cache.batch_size() != batch) {

        throw std::runtime_error(
            "MultiHeadAttention::forward_cached: "
            "input batch size does not match KV cache"
        );
    }

    // --------------------------------------------------------
    // Old sequence before append.
    // --------------------------------------------------------

    const std::size_t old_sequence =
        cache.initialized()
            ? cache.sequence_length()
            : 0;

    // --------------------------------------------------------
    // Project Q/K/V.
    // --------------------------------------------------------

    Tensor q_all;

    Tensor k_new;

    Tensor v_new;

    Tensor result =
        Tensor::zeros(
            input.shape(),
            DType::Float32,
            input.device()
        );

    const float* input_data =
        input.data_as<float>();

    float* result_data =
        result.data_as<float>();

    const std::size_t new_sequence_elements =
        new_sequence *
        embed_dim_;

    // --------------------------------------------------------
    // Process each batch independently.
    // --------------------------------------------------------

    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {

        Tensor sequence =
            Tensor::zeros(
                {new_sequence, embed_dim_},
                DType::Float32,
                input.device()
            );

        float* sequence_data =
            sequence.data_as<float>();

        for (std::size_t i = 0;
             i < new_sequence_elements;
             ++i) {

            sequence_data[i] =
                input_data[
                    batch_index *
                    new_sequence_elements +
                    i
                ];
        }

        Tensor q =
            project_2d(
                sequence,
                q_weight_,
                q_bias_,
                use_bias_
            );

        Tensor k =
            project_2d(
                sequence,
                k_weight_,
                k_bias_,
                use_bias_
            );

        Tensor v =
            project_2d(
                sequence,
                v_weight_,
                v_bias_,
                use_bias_
            );

        // ----------------------------------------------------
        // For 2D cache, append directly.
        // ----------------------------------------------------

        if (input.ndim() == 2) {

            cache.append(
                k,
                v
            );

            const Tensor& cached_key =
                cache.key();

            const Tensor& cached_value =
                cache.value();

            Tensor context =
                cached_attention_2d(
                    q,
                    cached_key,
                    cached_value,
                    num_heads_,
                    causal_,
                    old_sequence
                );

            Tensor projected =
                project_2d(
                    context,
                    out_weight_,
                    out_bias_,
                    use_bias_
                );

            const float* projected_data =
                projected.data_as<float>();

            for (std::size_t i = 0;
                 i < new_sequence_elements;
                 ++i) {

                result_data[i] =
                    projected_data[i];
            }

            break;
        }

        // ----------------------------------------------------
        // 3D cache.
        //
        // KVCache is batched, therefore append the complete
        // batch together. We first construct the full K/V
        // blocks and append once.
        // ----------------------------------------------------

        if (batch_index == 0) {

            Tensor full_k =
                Tensor::zeros(
                    {
                        batch,
                        new_sequence,
                        embed_dim_
                    },
                    DType::Float32,
                    input.device()
                );

            Tensor full_v =
                Tensor::zeros(
                    {
                        batch,
                        new_sequence,
                        embed_dim_
                    },
                    DType::Float32,
                    input.device()
                );

            float* full_k_data =
                full_k.data_as<float>();

            float* full_v_data =
                full_v.data_as<float>();

            // ----------------------------------------------
            // Fill current batch and remaining batches.
            // ----------------------------------------------

            for (std::size_t b = 0;
                 b < batch;
                 ++b) {

                Tensor batch_sequence =
                    Tensor::zeros(
                        {
                            new_sequence,
                            embed_dim_
                        },
                        DType::Float32,
                        input.device()
                    );

                float* batch_sequence_data =
                    batch_sequence.data_as<float>();

                for (std::size_t i = 0;
                     i < new_sequence_elements;
                     ++i) {

                    batch_sequence_data[i] =
                        input_data[
                            b *
                            new_sequence_elements +
                            i
                        ];
                }

                Tensor batch_k =
                    project_2d(
                        batch_sequence,
                        k_weight_,
                        k_bias_,
                        use_bias_
                    );

                Tensor batch_v =
                    project_2d(
                        batch_sequence,
                        v_weight_,
                        v_bias_,
                        use_bias_
                    );

                const float* batch_k_data =
                    batch_k.data_as<float>();

                const float* batch_v_data =
                    batch_v.data_as<float>();

                for (std::size_t i = 0;
                     i < new_sequence_elements;
                     ++i) {

                    full_k_data[
                        b *
                        new_sequence_elements +
                        i
                    ] =
                        batch_k_data[i];

                    full_v_data[
                        b *
                        new_sequence_elements +
                        i
                    ] =
                        batch_v_data[i];
                }
            }

            cache.append(
                full_k,
                full_v
            );
        }

        // ----------------------------------------------------
        // Retrieve the complete cache.
        // ----------------------------------------------------

        const Tensor& cached_key =
            cache.key();

        const Tensor& cached_value =
            cache.value();

        const float* cached_key_data =
            cached_key.data_as<float>();

        const float* cached_value_data =
            cached_value.data_as<float>();

        const std::size_t total_sequence =
            cache.sequence_length();

        Tensor batch_cached_key =
            Tensor::zeros(
                {
                    total_sequence,
                    embed_dim_
                },
                DType::Float32,
                input.device()
            );

        Tensor batch_cached_value =
            Tensor::zeros(
                {
                    total_sequence,
                    embed_dim_
                },
                DType::Float32,
                input.device()
            );

        float* batch_cached_key_data =
            batch_cached_key.data_as<float>();

        float* batch_cached_value_data =
            batch_cached_value.data_as<float>();

        const std::size_t total_elements =
            total_sequence *
            embed_dim_;

        for (std::size_t i = 0;
             i < total_elements;
             ++i) {

            batch_cached_key_data[i] =
                cached_key_data[
                    batch_index *
                    total_elements +
                    i
                ];

            batch_cached_value_data[i] =
                cached_value_data[
                    batch_index *
                    total_elements +
                    i
                ];
        }

        Tensor context =
            cached_attention_2d(
                q,
                batch_cached_key,
                batch_cached_value,
                num_heads_,
                causal_,
                old_sequence
            );

        Tensor projected =
            project_2d(
                context,
                out_weight_,
                out_bias_,
                use_bias_
            );

        const float* projected_data =
            projected.data_as<float>();

        for (std::size_t i = 0;
             i < new_sequence_elements;
             ++i) {

            result_data[
                batch_index *
                new_sequence_elements +
                i
            ] =
                projected_data[i];
        }
    }

    return result;
}

// ============================================================
// PARAMETERS
// ============================================================

const Tensor&
MultiHeadAttention::q_weight() const {
    return q_weight_;
}

Tensor&
MultiHeadAttention::q_weight() {
    return q_weight_;
}

const Tensor&
MultiHeadAttention::k_weight() const {
    return k_weight_;
}

Tensor&
MultiHeadAttention::k_weight() {
    return k_weight_;
}

const Tensor&
MultiHeadAttention::v_weight() const {
    return v_weight_;
}

Tensor&
MultiHeadAttention::v_weight() {
    return v_weight_;
}

const Tensor&
MultiHeadAttention::out_weight() const {
    return out_weight_;
}

Tensor&
MultiHeadAttention::out_weight() {
    return out_weight_;
}

const Tensor&
MultiHeadAttention::q_bias() const {
    return q_bias_;
}

Tensor&
MultiHeadAttention::q_bias() {
    return q_bias_;
}

const Tensor&
MultiHeadAttention::k_bias() const {
    return k_bias_;
}

Tensor&
MultiHeadAttention::k_bias() {
    return k_bias_;
}

const Tensor&
MultiHeadAttention::v_bias() const {
    return v_bias_;
}

Tensor&
MultiHeadAttention::v_bias() {
    return v_bias_;
}

const Tensor&
MultiHeadAttention::out_bias() const {
    return out_bias_;
}

Tensor&
MultiHeadAttention::out_bias() {
    return out_bias_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
MultiHeadAttention::embed_dim() const {
    return embed_dim_;
}

std::size_t
MultiHeadAttention::num_heads() const {
    return num_heads_;
}

std::size_t
MultiHeadAttention::head_dim() const {
    return head_dim_;
}

bool
MultiHeadAttention::has_bias() const {
    return use_bias_;
}

bool
MultiHeadAttention::is_causal() const {
    return causal_;
}

void
MultiHeadAttention::set_causal(
    bool enabled
) {
    causal_ = enabled;
}

} // namespace venla
