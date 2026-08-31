#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "venla/core/device.hpp"
#include "venla/core/dtype.hpp"
#include "venla/core/shape.hpp"
#include "venla/nn/cross_entropy_loss.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/tensor/tensor.hpp"
#include "venla/tokenizer/tokenizer.hpp"
#include "venla/training/causal_lm.hpp"
#include "venla/training/trainer.hpp"

#include <cstdint>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <vector>

namespace py = pybind11;

namespace {

venla::Tensor tensor_from_int64(
    const std::vector<std::int64_t>& values
) {
    venla::Tensor tensor =
        venla::Tensor::empty(
            {values.size()},
            venla::DType::Int64,
            venla::Device::cpu()
        );

    auto* data =
        tensor.data_as<std::int64_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {
        data[i] = values[i];
    }

    return tensor;
}

venla::Tensor tensor_from_int32(
    const std::vector<std::int32_t>& values
) {
    venla::Tensor tensor =
        venla::Tensor::empty(
            {values.size()},
            venla::DType::Int32,
            venla::Device::cpu()
        );

    auto* data =
        tensor.data_as<std::int32_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {
        data[i] = values[i];
    }

    return tensor;
}

std::vector<std::int64_t> tensor_to_int64(
    const venla::Tensor& tensor
) {
    if (tensor.dtype() == venla::DType::Int64) {
        const auto* data =
            tensor.data_as<std::int64_t>();

        return std::vector<std::int64_t>(
            data,
            data + tensor.numel()
        );
    }

    if (tensor.dtype() == venla::DType::Int32) {
        const auto* data =
            tensor.data_as<std::int32_t>();

        std::vector<std::int64_t> result(
            tensor.numel()
        );

        for (std::size_t i = 0;
             i < tensor.numel();
             ++i) {
            result[i] =
                static_cast<std::int64_t>(
                    data[i]
                );
        }

        return result;
    }

    throw std::invalid_argument(
        "tensor_to_int64: tensor must use Int32 or Int64"
    );
}

} // namespace

// ============================================================
// VENLACPU MODEL WEIGHT PERSISTENCE
// ============================================================

namespace {

constexpr char VENLA_WEIGHTS_MAGIC[8] = {
    'V', 'N', 'L', 'W', '0', '0', '0', '1'
};

constexpr std::uint32_t VENLA_WEIGHTS_VERSION = 1;

template <typename T>
void write_binary(
    std::ofstream& file,
    const T& value
) {
    file.write(
        reinterpret_cast<const char*>(&value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!file) {
        throw std::runtime_error(
            "VENLACPU: gagal menulis file weights."
        );
    }
}

template <typename T>
void read_binary(
    std::ifstream& file,
    T& value
) {
    file.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(T))
    );

    if (!file) {
        throw std::runtime_error(
            "VENLACPU: file weights rusak atau tidak lengkap."
        );
    }
}

void save_model_weights(
    venla::LanguageModel& model,
    const std::string& path
) {
    std::ofstream file(
        path,
        std::ios::binary |
        std::ios::trunc
    );

    if (!file) {
        throw std::runtime_error(
            "VENLACPU: tidak dapat membuka file weights: " +
            path
        );
    }

    file.write(
        VENLA_WEIGHTS_MAGIC,
        sizeof(VENLA_WEIGHTS_MAGIC)
    );

    write_binary(
        file,
        VENLA_WEIGHTS_VERSION
    );

    const std::uint64_t vocab_size =
        static_cast<std::uint64_t>(
            model.vocab_size()
        );

    const std::uint64_t max_seq_len =
        static_cast<std::uint64_t>(
            model.max_seq_len()
        );

    const std::uint64_t embed_dim =
        static_cast<std::uint64_t>(
            model.embed_dim()
        );

    const std::uint64_t num_heads =
        static_cast<std::uint64_t>(
            model.num_heads()
        );

    const std::uint64_t hidden_dim =
        static_cast<std::uint64_t>(
            model.hidden_dim()
        );

    const std::uint64_t num_layers =
        static_cast<std::uint64_t>(
            model.num_layers()
        );

    const std::uint8_t use_bias =
        model.has_bias() ? 1 : 0;

    write_binary(file, vocab_size);
    write_binary(file, max_seq_len);
    write_binary(file, embed_dim);
    write_binary(file, num_heads);
    write_binary(file, hidden_dim);
    write_binary(file, num_layers);
    write_binary(file, use_bias);

    const std::vector<venla::Tensor*> parameters =
        model.parameters();

    const std::uint64_t parameter_count =
        static_cast<std::uint64_t>(
            parameters.size()
        );

    write_binary(
        file,
        parameter_count
    );

    for (
        std::size_t i = 0;
        i < parameters.size();
        ++i
    ) {
        venla::Tensor* tensor =
            parameters[i];

        if (tensor == nullptr) {
            throw std::runtime_error(
                "VENLACPU: parameter null."
            );
        }

        if (!tensor->is_contiguous()) {
            throw std::runtime_error(
                "VENLACPU: parameter harus contiguous."
            );
        }

        const std::uint32_t dtype =
            static_cast<std::uint32_t>(
                tensor->dtype()
            );

        const std::uint64_t ndim =
            static_cast<std::uint64_t>(
                tensor->ndim()
            );

        write_binary(
            file,
            dtype
        );

        write_binary(
            file,
            ndim
        );

        const auto& dimensions =
            tensor->shape().dimensions();

        for (
            std::size_t dimension : dimensions
        ) {
            const std::uint64_t value =
                static_cast<std::uint64_t>(
                    dimension
                );

            write_binary(
                file,
                value
            );
        }

        const std::uint64_t nbytes =
            static_cast<std::uint64_t>(
                tensor->nbytes()
            );

        write_binary(
            file,
            nbytes
        );

        if (nbytes > 0) {
            file.write(
                reinterpret_cast<const char*>(
                    tensor->data()
                ),
                static_cast<std::streamsize>(
                    nbytes
                )
            );

            if (!file) {
                throw std::runtime_error(
                    "VENLACPU: gagal menulis parameter."
                );
            }
        }
    }
}

void load_model_weights(
    venla::LanguageModel& model,
    const std::string& path
) {
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "VENLACPU: tidak dapat membuka file weights: " +
            path
        );
    }

    char magic[
        sizeof(VENLA_WEIGHTS_MAGIC)
    ];

    file.read(
        magic,
        sizeof(magic)
    );

    if (
        std::memcmp(
            magic,
            VENLA_WEIGHTS_MAGIC,
            sizeof(VENLA_WEIGHTS_MAGIC)
        ) != 0
    ) {
        throw std::runtime_error(
            "VENLACPU: magic weights tidak valid."
        );
    }

    std::uint32_t version = 0;

    read_binary(
        file,
        version
    );

    if (
        version !=
        VENLA_WEIGHTS_VERSION
    ) {
        throw std::runtime_error(
            "VENLACPU: versi format weights tidak didukung."
        );
    }

    std::uint64_t vocab_size = 0;
    std::uint64_t max_seq_len = 0;
    std::uint64_t embed_dim = 0;
    std::uint64_t num_heads = 0;
    std::uint64_t hidden_dim = 0;
    std::uint64_t num_layers = 0;
    std::uint8_t use_bias = 0;
    std::uint64_t parameter_count = 0;

    read_binary(file, vocab_size);
    read_binary(file, max_seq_len);
    read_binary(file, embed_dim);
    read_binary(file, num_heads);
    read_binary(file, hidden_dim);
    read_binary(file, num_layers);
    read_binary(file, use_bias);
    read_binary(file, parameter_count);

    if (
        vocab_size != model.vocab_size() ||
        max_seq_len != model.max_seq_len() ||
        embed_dim != model.embed_dim() ||
        num_heads != model.num_heads() ||
        hidden_dim != model.hidden_dim() ||
        num_layers != model.num_layers() ||
        use_bias != (
            model.has_bias() ? 1 : 0
        )
    ) {
        throw std::runtime_error(
            "VENLACPU: arsitektur model berbeda "
            "dengan file weights."
        );
    }

    const std::vector<venla::Tensor*> parameters =
        model.parameters();

    if (
        parameter_count !=
        parameters.size()
    ) {
        throw std::runtime_error(
            "VENLACPU: jumlah parameter berbeda."
        );
    }

    for (
        std::size_t i = 0;
        i < parameters.size();
        ++i
    ) {
        venla::Tensor* tensor =
            parameters[i];

        std::uint32_t dtype = 0;
        std::uint64_t ndim = 0;

        read_binary(
            file,
            dtype
        );

        read_binary(
            file,
            ndim
        );

        std::vector<std::size_t> dimensions(
            static_cast<std::size_t>(ndim)
        );

        for (
            std::size_t d = 0;
            d < dimensions.size();
            ++d
        ) {
            std::uint64_t dimension = 0;

            read_binary(
                file,
                dimension
            );

            dimensions[d] =
                static_cast<std::size_t>(
                    dimension
                );
        }

        std::uint64_t nbytes = 0;

        read_binary(
            file,
            nbytes
        );

        if (
            static_cast<std::uint32_t>(
                tensor->dtype()
            ) != dtype
        ) {
            throw std::runtime_error(
                "VENLACPU: dtype parameter berbeda."
            );
        }

        if (
            tensor->shape().dimensions()
            != dimensions
        ) {
            throw std::runtime_error(
                "VENLACPU: shape parameter berbeda."
            );
        }

        if (
            tensor->nbytes()
            != static_cast<std::size_t>(
                nbytes
            )
        ) {
            throw std::runtime_error(
                "VENLACPU: ukuran parameter berbeda."
            );
        }

        if (!tensor->is_contiguous()) {
            throw std::runtime_error(
                "VENLACPU: parameter harus contiguous."
            );
        }

        if (nbytes > 0) {
            file.read(
                reinterpret_cast<char*>(
                    tensor->data()
                ),
                static_cast<std::streamsize>(
                    nbytes
                )
            );

            if (!file) {
                throw std::runtime_error(
                    "VENLACPU: gagal membaca parameter."
                );
            }
        }
    }
}

} // namespace


PYBIND11_MODULE(_venlacpu, m) {

    m.doc() =
        "VENLACPU native C++17 CPU-first deep learning engine";

    // ========================================================
    // VERSION
    // ========================================================

    m.attr("__version__") = "1.0.0";

    // ========================================================
    // DEVICE
    // ========================================================

    py::enum_<venla::DeviceType>(
        m,
        "DeviceType"
    )
        .value(
            "CPU",
            venla::DeviceType::CPU
        )
        .export_values();

    py::class_<venla::Device>(
        m,
        "Device"
    )
        .def(
            py::init<>()
        )
        .def(
            py::init<venla::DeviceType>()
        )
        .def_static(
            "cpu",
            &venla::Device::cpu
        )
        .def(
            "type",
            &venla::Device::type
        )
        .def(
            "is_cpu",
            &venla::Device::is_cpu
        )
        .def(
            "__str__",
            &venla::Device::to_string
        )
        .def(
            "__repr__",
            &venla::Device::to_string
        );

    // ========================================================
    // DTYPE
    // ========================================================

    py::enum_<venla::DType>(
        m,
        "DType"
    )
        .value("Bool", venla::DType::Bool)
        .value("Int8", venla::DType::Int8)
        .value("Int16", venla::DType::Int16)
        .value("Int32", venla::DType::Int32)
        .value("Int64", venla::DType::Int64)
        .value("UInt8", venla::DType::UInt8)
        .value("UInt16", venla::DType::UInt16)
        .value("UInt32", venla::DType::UInt32)
        .value("UInt64", venla::DType::UInt64)
        .value("Float16", venla::DType::Float16)
        .value("BFloat16", venla::DType::BFloat16)
        .value("Float32", venla::DType::Float32)
        .value("Float64", venla::DType::Float64)
        .value("Complex64", venla::DType::Complex64)
        .value("Complex128", venla::DType::Complex128)
        .export_values();

    m.def(
        "dtype_name",
        &venla::dtype_name
    );

    m.def(
        "dtype_size",
        &venla::dtype_size
    );

    m.def(
        "dtype_is_floating",
        &venla::dtype_is_floating
    );

    m.def(
        "dtype_is_integral",
        &venla::dtype_is_integral
    );

    m.def(
        "dtype_is_complex",
        &venla::dtype_is_complex
    );

    // ========================================================
    // SHAPE
    // ========================================================

    py::class_<venla::Shape>(
        m,
        "Shape"
    )
        .def(
            py::init<>()
        )
        .def(
            py::init<
                const std::vector<std::size_t>&
            >()
        )
        .def(
            "ndim",
            &venla::Shape::ndim
        )
        .def(
            "rank",
            &venla::Shape::rank
        )
        .def(
            "numel",
            &venla::Shape::numel
        )
        .def(
            "dimensions",
            &venla::Shape::dimensions,
            py::return_value_policy::reference_internal
        )
        .def(
            "__getitem__",
            &venla::Shape::operator[]
        )
        .def(
            "__eq__",
            &venla::Shape::operator==
        )
        .def(
            "__ne__",
            &venla::Shape::operator!=
        )
        .def(
            "__str__",
            &venla::Shape::to_string
        )
        .def(
            "__repr__",
            &venla::Shape::to_string
        );

    // ========================================================
    // TENSOR
    // ========================================================

    py::class_<venla::Tensor>(
        m,
        "Tensor"
    )
        .def(
            py::init<>()
        )
        .def(
            py::init<
                const venla::Shape&,
                venla::DType,
                const venla::Device&
            >(),
            py::arg("shape"),
            py::arg("dtype") = venla::DType::Float32,
            py::arg("device") = venla::Device::cpu()
        )
        .def_static(
            "zeros",
            [](const venla::Shape& shape,
               venla::DType dtype,
               const venla::Device& device) {
                return venla::Tensor::zeros(
                    shape,
                    dtype,
                    device
                );
            },
            py::arg("shape"),
            py::arg("dtype") = venla::DType::Float32,
            py::arg("device") = venla::Device::cpu()
        )
        .def_static(
            "ones",
            [](const venla::Shape& shape,
               venla::DType dtype,
               const venla::Device& device) {
                return venla::Tensor::ones(
                    shape,
                    dtype,
                    device
                );
            },
            py::arg("shape"),
            py::arg("dtype") = venla::DType::Float32,
            py::arg("device") = venla::Device::cpu()
        )
        .def_static(
            "empty",
            [](const venla::Shape& shape,
               venla::DType dtype,
               const venla::Device& device) {
                return venla::Tensor::empty(
                    shape,
                    dtype,
                    device
                );
            },
            py::arg("shape"),
            py::arg("dtype") = venla::DType::Float32,
            py::arg("device") = venla::Device::cpu()
        )
        .def(
            "requires_grad_",
            &venla::Tensor::requires_grad_,
            py::arg("enabled") = true
        )
        .def(
            "requires_grad",
            &venla::Tensor::requires_grad
        )
        .def(
            "is_leaf",
            &venla::Tensor::is_leaf
        )
        .def(
            "has_grad",
            &venla::Tensor::has_grad
        )
        .def(
            "zero_grad",
            &venla::Tensor::zero_grad
        )
        .def(
            "backward",
            py::overload_cast<>(
                &venla::Tensor::backward
            )
        )
        .def(
            "backward",
            py::overload_cast<
                const venla::Tensor&
            >(
                &venla::Tensor::backward
            )
        )
        .def(
            "shape",
            &venla::Tensor::shape,
            py::return_value_policy::reference_internal
        )
        .def(
            "stride",
            &venla::Tensor::stride,
            py::return_value_policy::reference_internal
        )
        .def(
            "dtype",
            &venla::Tensor::dtype
        )
        .def(
            "device",
            &venla::Tensor::device,
            py::return_value_policy::reference_internal
        )
        .def(
            "ndim",
            &venla::Tensor::ndim
        )
        .def(
            "rank",
            &venla::Tensor::rank
        )
        .def(
            "numel",
            &venla::Tensor::numel
        )
        .def(
            "nbytes",
            &venla::Tensor::nbytes
        )
        .def(
            "is_contiguous",
            &venla::Tensor::is_contiguous
        )
        .def(
            "is_empty",
            py::overload_cast<>(
                &venla::Tensor::empty,
                py::const_
            )
        )
        .def(
            "info",
            &venla::Tensor::info
        )
        .def(
            "__repr__",
            &venla::Tensor::info
        )
        .def(
            "__str__",
            &venla::Tensor::info
        );

    // ========================================================
    // SIMPLE TENSOR HELPERS
    // ========================================================

    m.def(
        "tensor",
        &tensor_from_int64
    );

    m.def(
        "tensor_int32",
        &tensor_from_int32
    );

    m.def(
        "tensor_to_int64",
        &tensor_to_int64
    );

    // ========================================================
    // BPE TOKENIZER
    // ========================================================

    py::class_<venla::BPETokenizer::Config>(
        m,
        "BPETokenizerConfig"
    )
        .def(
            py::init<>()
        )
        .def_readwrite(
            "vocab_size",
            &venla::BPETokenizer::Config::vocab_size
        )
        .def_readwrite(
            "min_frequency",
            &venla::BPETokenizer::Config::min_frequency
        )
        .def_readwrite(
            "add_bos",
            &venla::BPETokenizer::Config::add_bos
        )
        .def_readwrite(
            "add_eos",
            &venla::BPETokenizer::Config::add_eos
        );

    py::class_<venla::BPETokenizer>(
        m,
        "BPETokenizer"
    )
        .def(
            py::init<>()
        )
        .def(
            py::init<
                const venla::BPETokenizer::Config&
            >(),
            py::arg("config")
        )
        .def(
            "train",
            py::overload_cast<
                const std::string&
            >(
                &venla::BPETokenizer::train
            ),
            py::arg("corpus")
        )
        .def(
            "train_documents",
            py::overload_cast<
                const std::vector<std::string>&
            >(
                &venla::BPETokenizer::train
            ),
            py::arg("documents")
        )
        .def(
            "encode",
            py::overload_cast<
                const std::string&
            >(
                &venla::BPETokenizer::encode,
                py::const_
            ),
            py::arg("text")
        )
        .def(
            "encode",
            py::overload_cast<
                const std::string&,
                bool,
                bool
            >(
                &venla::BPETokenizer::encode,
                py::const_
            ),
            py::arg("text"),
            py::arg("add_bos"),
            py::arg("add_eos")
        )
        .def(
            "decode",
            &venla::BPETokenizer::decode,
            py::arg("ids")
        )
        .def(
            "save",
            &venla::BPETokenizer::save,
            py::arg("path")
        )
        .def_static(
            "load",
            &venla::BPETokenizer::load,
            py::arg("path")
        )
        .def(
            "trained",
            &venla::BPETokenizer::trained
        )
        .def(
            "merge_count",
            &venla::BPETokenizer::merge_count
        )
        .def(
            "vocab_size",
            [](const venla::BPETokenizer& tokenizer) {
                return tokenizer.vocabulary().tokens().size();
            }
        )
        .def(
            "vocabulary",
            [](const venla::BPETokenizer& tokenizer) {
                return tokenizer.vocabulary().tokens();
            }
        );

    m.attr("Tokenizer") =
        m.attr("BPETokenizer");

    // ========================================================
    // LANGUAGE MODEL
    // ========================================================

    py::class_<venla::LanguageModel>(
        m,
        "LanguageModel"
    )
        .def(
            py::init<
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                bool
            >(),
            py::arg("vocab_size"),
            py::arg("max_seq_len"),
            py::arg("embed_dim"),
            py::arg("num_heads"),
            py::arg("hidden_dim"),
            py::arg("num_layers"),
            py::arg("use_bias") = true
        )
        .def(
            "forward",
            &venla::LanguageModel::forward
        )
        .def(
            "parameters",
            &venla::LanguageModel::parameters,
            py::return_value_policy::reference
        )
        .def(
            "vocab_size",
            &venla::LanguageModel::vocab_size
        )
        .def(
            "max_seq_len",
            &venla::LanguageModel::max_seq_len
        )
        .def(
            "embed_dim",
            &venla::LanguageModel::embed_dim
        )
        .def(
            "num_heads",
            &venla::LanguageModel::num_heads
        )
        .def(
            "hidden_dim",
            &venla::LanguageModel::hidden_dim
        )
        .def(
            "num_layers",
            &venla::LanguageModel::num_layers
        )
        .def(
            "has_bias",
            &venla::LanguageModel::has_bias
        )
        .def(
            "save_weights",
            [](
                venla::LanguageModel& model,
                const std::string& path
            ) {
                save_model_weights(
                    model,
                    path
                );
            },
            py::arg("path")
        )
        .def(
            "load_weights",
            [](
                venla::LanguageModel& model,
                const std::string& path
            ) {
                load_model_weights(
                    model,
                    path
                );
            },
            py::arg("path")
        );

    // ========================================================
    // OPTIMIZER
    // ========================================================

    py::class_<venla::Optimizer>(
        m,
        "Optimizer"
    )
        .def(
            "add_parameter",
            &venla::Optimizer::add_parameter
        )
        .def(
            "add_parameters",
            &venla::Optimizer::add_parameters
        )
        .def(
            "parameter_count",
            &venla::Optimizer::parameter_count
        )
        .def(
            "zero_grad",
            &venla::Optimizer::zero_grad
        );

    py::class_<venla::SGD, venla::Optimizer>(
        m,
        "SGD"
    )
        .def(
            py::init<float, float, float>(),
            py::arg("learning_rate") = 0.001f,
            py::arg("momentum") = 0.0f,
            py::arg("weight_decay") = 0.0f
        )
        .def(
            "step",
            &venla::SGD::step
        )
        .def(
            "learning_rate",
            &venla::SGD::learning_rate
        )
        .def(
            "momentum",
            &venla::SGD::momentum
        )
        .def(
            "weight_decay",
            &venla::SGD::weight_decay
        )
        .def(
            "set_learning_rate",
            &venla::SGD::set_learning_rate
        );

    py::class_<venla::Adam, venla::Optimizer>(
        m,
        "Adam"
    )
        .def(
            py::init<
                float,
                float,
                float,
                float,
                float
            >(),
            py::arg("learning_rate") = 0.001f,
            py::arg("beta1") = 0.9f,
            py::arg("beta2") = 0.999f,
            py::arg("epsilon") = 1e-8f,
            py::arg("weight_decay") = 0.0f
        )
        .def(
            "step",
            &venla::Adam::step
        )
        .def(
            "learning_rate",
            &venla::Adam::learning_rate
        )
        .def(
            "beta1",
            &venla::Adam::beta1
        )
        .def(
            "beta2",
            &venla::Adam::beta2
        )
        .def(
            "epsilon",
            &venla::Adam::epsilon
        )
        .def(
            "weight_decay",
            &venla::Adam::weight_decay
        )
        .def(
            "step_count",
            &venla::Adam::step_count
        )
        .def(
            "set_learning_rate",
            &venla::Adam::set_learning_rate
        );

    // ========================================================
    // CAUSAL LM BATCH
    // ========================================================

    py::class_<venla::CausalLMBatch>(
        m,
        "CausalLMBatch"
    )
        .def_readonly(
            "input",
            &venla::CausalLMBatch::input
        )
        .def_readonly(
            "targets",
            &venla::CausalLMBatch::targets
        )
        .def_readonly(
            "batch_size",
            &venla::CausalLMBatch::batch_size
        )
        .def_readonly(
            "sequence_length",
            &venla::CausalLMBatch::sequence_length
        )
        .def_readonly(
            "valid_tokens",
            &venla::CausalLMBatch::valid_tokens
        );

    // ========================================================
    // CAUSAL LM DATASET
    // ========================================================

    py::class_<venla::CausalLMDataset>(
        m,
        "CausalLMDataset"
    )
        .def(
            py::init<
                std::int64_t,
                std::int64_t
            >(),
            py::arg("pad_token_id") = 0,
            py::arg("ignore_index") = -100
        )
        .def(
            "add_sequence",
            &venla::CausalLMDataset::add_sequence
        )
        .def(
            "add_sequence_int32",
            &venla::CausalLMDataset::add_sequence_int32
        )
        .def(
            "clear",
            &venla::CausalLMDataset::clear
        )
        .def(
            "size",
            &venla::CausalLMDataset::size
        )
        .def(
            "empty",
            &venla::CausalLMDataset::empty
        )
        .def(
            "max_sequence_length",
            &venla::CausalLMDataset::max_sequence_length
        )
        .def(
            "num_batches",
            &venla::CausalLMDataset::num_batches,
            py::arg("batch_size"),
            py::arg("drop_last") = false
        )
        .def(
            "batch",
            &venla::CausalLMDataset::batch,
            py::arg("batch_index"),
            py::arg("batch_size"),
            py::arg("drop_last") = false
        )
        .def(
            "pad_token_id",
            &venla::CausalLMDataset::pad_token_id
        )
        .def(
            "ignore_index",
            &venla::CausalLMDataset::ignore_index
        );

    m.def(
        "make_causal_lm_batch",
        &venla::make_causal_lm_batch,
        py::arg("tokens"),
        py::arg("ignore_index") = -100
    );

    // ========================================================
    // TRAINER CONFIG
    // ========================================================

    py::class_<venla::TrainerConfig>(
        m,
        "TrainerConfig"
    )
        .def(
            py::init<>()
        )
        .def_readwrite(
            "epochs",
            &venla::TrainerConfig::epochs
        )
        .def_readwrite(
            "batch_size",
            &venla::TrainerConfig::batch_size
        )
        .def_readwrite(
            "gradient_accumulation_steps",
            &venla::TrainerConfig::gradient_accumulation_steps
        )
        .def_readwrite(
            "drop_last",
            &venla::TrainerConfig::drop_last
        )
        .def_readwrite(
            "log_every",
            &venla::TrainerConfig::log_every
        )
        .def_readwrite(
            "ignore_index",
            &venla::TrainerConfig::ignore_index
        );

    // ========================================================
    // TRAINING METRICS
    // ========================================================

    py::class_<venla::TrainingMetrics>(
        m,
        "TrainingMetrics"
    )
        .def(
            py::init<>()
        )
        .def_readonly(
            "loss",
            &venla::TrainingMetrics::loss
        )
        .def_readonly(
            "tokens",
            &venla::TrainingMetrics::tokens
        )
        .def_readonly(
            "batches",
            &venla::TrainingMetrics::batches
        )
        .def_readonly(
            "optimizer_steps",
            &venla::TrainingMetrics::optimizer_steps
        )
        .def_readonly(
            "epoch",
            &venla::TrainingMetrics::epoch
        )
        .def_readonly(
            "global_step",
            &venla::TrainingMetrics::global_step
        );

    // ========================================================
    // TRAINER
    // ========================================================

    py::class_<venla::Trainer>(
        m,
        "Trainer"
    )
        .def(
            py::init<
                venla::LanguageModel&,
                venla::Optimizer&,
                const venla::TrainerConfig&
            >(),
            py::arg("model"),
            py::arg("optimizer"),
            py::arg("config") = venla::TrainerConfig()
        )
        .def(
            "train_epoch",
            &venla::Trainer::train_epoch
        )
        .def(
            "fit",
            &venla::Trainer::fit
        )
        .def(
            "evaluate",
            &venla::Trainer::evaluate
        )
        .def(
            "current_epoch",
            &venla::Trainer::current_epoch
        )
        .def(
            "global_step",
            &venla::Trainer::global_step
        )
        .def(
            "last_metrics",
            &venla::Trainer::last_metrics,
            py::return_value_policy::reference_internal
        )
        .def(
            "model",
            &venla::Trainer::model,
            py::return_value_policy::reference
        )
        .def(
            "optimizer",
            &venla::Trainer::optimizer,
            py::return_value_policy::reference
        )
        .def(
            "config",
            &venla::Trainer::config,
            py::return_value_policy::reference_internal
        );

    // ========================================================
    // GENERATION CONFIG
    // ========================================================

    py::class_<venla::GenerationConfig>(
        m,
        "GenerationConfig"
    )
        .def(
            py::init<>()
        )
        .def_readwrite(
            "max_new_tokens",
            &venla::GenerationConfig::max_new_tokens
        )
        .def_readwrite(
            "temperature",
            &venla::GenerationConfig::temperature
        )
        .def_readwrite(
            "eos_token_id",
            &venla::GenerationConfig::eos_token_id
        )
        .def_readwrite(
            "stop_on_eos",
            &venla::GenerationConfig::stop_on_eos
        );

    m.def(
        "generate",
        &venla::generate,
        py::arg("model"),
        py::arg("prompt"),
        py::arg("config") = venla::GenerationConfig()
    );

}
