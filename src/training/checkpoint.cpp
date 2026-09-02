#include "venla/training/checkpoint.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

constexpr char MAGIC[] = "VNLCP230";
constexpr std::uint32_t FORMAT_VERSION = 1;

template <typename T>
void write_value(
    std::ostream& stream,
    const T& value
) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        sizeof(T)
    );

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint write failed"
        );
    }
}

template <typename T>
T read_value(
    std::istream& stream
) {
    T value{};

    stream.read(
        reinterpret_cast<char*>(&value),
        sizeof(T)
    );

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint read failed"
        );
    }

    return value;
}

void write_tensor(
    std::ostream& stream,
    const Tensor& tensor
) {
    const std::uint64_t ndim =
        static_cast<std::uint64_t>(
            tensor.ndim()
        );

    write_value(stream, ndim);

    for (std::size_t i = 0; i < tensor.ndim(); ++i) {
        const std::uint64_t dim =
            static_cast<std::uint64_t>(
                tensor.shape()[i]
            );

        write_value(stream, dim);
    }

    const std::uint64_t numel =
        static_cast<std::uint64_t>(
            tensor.numel()
        );

    write_value(stream, numel);

    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(
            tensor.nbytes()
        );

    write_value(stream, nbytes);

    if (nbytes > 0) {
        stream.write(
            reinterpret_cast<const char*>(
                tensor.data()
            ),
            static_cast<std::streamsize>(
                nbytes
            )
        );
    }

    if (!stream) {
        throw std::runtime_error(
            "Failed to write model tensor"
        );
    }
}

void read_tensor(
    std::istream& stream,
    Tensor& tensor
) {
    const std::uint64_t ndim =
        read_value<std::uint64_t>(stream);

    if (
        ndim !=
        static_cast<std::uint64_t>(tensor.ndim())
    ) {
        throw std::runtime_error(
            "Checkpoint tensor rank mismatch"
        );
    }

    for (std::size_t i = 0; i < tensor.ndim(); ++i) {
        const std::uint64_t dim =
            read_value<std::uint64_t>(stream);

        if (
            dim !=
            static_cast<std::uint64_t>(
                tensor.shape()[i]
            )
        ) {
            throw std::runtime_error(
                "Checkpoint tensor shape mismatch"
            );
        }
    }

    const std::uint64_t numel =
        read_value<std::uint64_t>(stream);

    const std::uint64_t nbytes =
        read_value<std::uint64_t>(stream);

    if (
        numel !=
        static_cast<std::uint64_t>(
            tensor.numel()
        )
    ) {
        throw std::runtime_error(
            "Checkpoint tensor element count mismatch"
        );
    }

    if (
        nbytes !=
        static_cast<std::uint64_t>(
            tensor.nbytes()
        )
    ) {
        throw std::runtime_error(
            "Checkpoint tensor byte size mismatch"
        );
    }

    if (nbytes > 0) {
        stream.read(
            reinterpret_cast<char*>(
                tensor.data()
            ),
            static_cast<std::streamsize>(
                nbytes
            )
        );
    }

    if (!stream) {
        throw std::runtime_error(
            "Failed to read model tensor"
        );
    }
}

void write_string(
    std::ostream& stream,
    const std::string& value
) {
    const std::uint64_t size =
        static_cast<std::uint64_t>(
            value.size()
        );

    write_value(stream, size);

    if (size > 0) {
        stream.write(
            value.data(),
            static_cast<std::streamsize>(size)
        );
    }

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint string write failed"
        );
    }
}

std::string read_string(
    std::istream& stream
) {
    const std::uint64_t size =
        read_value<std::uint64_t>(stream);

    if (
        size >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "Checkpoint string size overflow"
        );
    }

    std::string value(
        static_cast<std::size_t>(size),
        '\0'
    );

    if (size > 0) {
        stream.read(
            value.data(),
            static_cast<std::streamsize>(size)
        );
    }

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint string read failed"
        );
    }

    return value;
}

} // namespace

void CheckpointEngine::save(
    const std::string& path,
    const LanguageModel& model,
    const Optimizer& optimizer,
    const CheckpointMetadata& metadata
) {
    std::ofstream stream(
        path,
        std::ios::binary |
        std::ios::trunc
    );

    if (!stream) {
        throw std::runtime_error(
            "Cannot open checkpoint for writing: " +
            path
        );
    }

    stream.write(
        MAGIC,
        sizeof(MAGIC) - 1
    );

    write_value(
        stream,
        FORMAT_VERSION
    );

    // --------------------------------------------------------
    // MODEL ARCHITECTURE
    // --------------------------------------------------------

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.vocab_size()
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.max_seq_len()
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.embed_dim()
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.num_heads()
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.hidden_dim()
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            model.num_layers()
        )
    );

    write_value(
        stream,
        model.has_bias()
    );

    // --------------------------------------------------------
    // MODEL PARAMETERS
    // --------------------------------------------------------

    const std::vector<Tensor*> parameters =
        const_cast<LanguageModel&>(
            model
        ).parameters();

    write_value(
        stream,
        static_cast<std::uint64_t>(
            parameters.size()
        )
    );

    for (const Tensor* parameter : parameters) {
        if (parameter == nullptr) {
            throw std::runtime_error(
                "Null model parameter"
            );
        }

        write_tensor(
            stream,
            *parameter
        );
    }

    // --------------------------------------------------------
    // OPTIMIZER
    // --------------------------------------------------------

    write_string(
        stream,
        optimizer.type_name()
    );

    optimizer.save_state(stream);

    // --------------------------------------------------------
    // TRAINER STATE
    // --------------------------------------------------------

    write_value(
        stream,
        static_cast<std::uint64_t>(
            metadata.epoch
        )
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            metadata.global_step
        )
    );

    write_value(
        stream,
        metadata.best_eval_loss
    );

    write_value(
        stream,
        static_cast<std::uint64_t>(
            metadata.bad_epochs
        )
    );

    write_value(
        stream,
        metadata.has_best_model
    );

    if (!stream) {
        throw std::runtime_error(
            "Checkpoint finalization failed"
        );
    }
}

CheckpointMetadata CheckpointEngine::load(
    const std::string& path,
    LanguageModel& model,
    Optimizer& optimizer
) {
    std::ifstream stream(
        path,
        std::ios::binary
    );

    if (!stream) {
        throw std::runtime_error(
            "Cannot open checkpoint for reading: " +
            path
        );
    }

    char magic[
        sizeof(MAGIC)
    ] = {};

    stream.read(
        magic,
        sizeof(MAGIC) - 1
    );

    if (
        std::memcmp(
            magic,
            MAGIC,
            sizeof(MAGIC) - 1
        ) != 0
    ) {
        throw std::runtime_error(
            "Invalid VENLACPU checkpoint magic"
        );
    }

    const std::uint32_t version =
        read_value<std::uint32_t>(
            stream
        );

    if (version != FORMAT_VERSION) {
        throw std::runtime_error(
            "Unsupported VENLACPU checkpoint version"
        );
    }

    // --------------------------------------------------------
    // MODEL ARCHITECTURE VALIDATION
    // --------------------------------------------------------

    const std::uint64_t vocab_size =
        read_value<std::uint64_t>(stream);

    const std::uint64_t max_seq_len =
        read_value<std::uint64_t>(stream);

    const std::uint64_t embed_dim =
        read_value<std::uint64_t>(stream);

    const std::uint64_t num_heads =
        read_value<std::uint64_t>(stream);

    const std::uint64_t hidden_dim =
        read_value<std::uint64_t>(stream);

    const std::uint64_t num_layers =
        read_value<std::uint64_t>(stream);

    const bool has_bias =
        read_value<bool>(stream);

    if (
        vocab_size != model.vocab_size() ||
        max_seq_len != model.max_seq_len() ||
        embed_dim != model.embed_dim() ||
        num_heads != model.num_heads() ||
        hidden_dim != model.hidden_dim() ||
        num_layers != model.num_layers() ||
        has_bias != model.has_bias()
    ) {
        throw std::runtime_error(
            "Checkpoint model architecture mismatch"
        );
    }

    // --------------------------------------------------------
    // MODEL PARAMETERS
    // --------------------------------------------------------

    const std::uint64_t parameter_count =
        read_value<std::uint64_t>(stream);

    std::vector<Tensor*> parameters =
        model.parameters();

    if (
        parameter_count !=
        static_cast<std::uint64_t>(
            parameters.size()
        )
    ) {
        throw std::runtime_error(
            "Checkpoint model parameter count mismatch"
        );
    }

    for (Tensor* parameter : parameters) {
        if (parameter == nullptr) {
            throw std::runtime_error(
                "Null model parameter"
            );
        }

        read_tensor(
            stream,
            *parameter
        );
    }

    // --------------------------------------------------------
    // OPTIMIZER
    // --------------------------------------------------------

    const std::string optimizer_type =
        read_string(stream);

    if (
        optimizer_type !=
        optimizer.type_name()
    ) {
        throw std::runtime_error(
            "Checkpoint optimizer type mismatch: " +
            optimizer_type +
            " vs " +
            optimizer.type_name()
        );
    }

    optimizer.load_state(stream);

    // --------------------------------------------------------
    // TRAINER STATE
    // --------------------------------------------------------

    CheckpointMetadata metadata;

    metadata.epoch =
        static_cast<std::size_t>(
            read_value<std::uint64_t>(stream)
        );

    metadata.global_step =
        static_cast<std::size_t>(
            read_value<std::uint64_t>(stream)
        );

    metadata.best_eval_loss =
        read_value<float>(stream);

    metadata.bad_epochs =
        static_cast<std::size_t>(
            read_value<std::uint64_t>(stream)
        );

    metadata.has_best_model =
        read_value<bool>(stream);

    return metadata;
}

} // namespace venla
