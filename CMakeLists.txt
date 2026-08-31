cmake_minimum_required(VERSION 3.16)

project(VENLACPU
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VENLA_BUILD_TESTS "Build VENLACPU tests" ON)
option(VENLA_BUILD_EXAMPLES "Build VENLACPU examples" ON)

# ============================================================
# VENLACPU CORE
# ============================================================

add_library(venlacpu_core STATIC

    src/core/version.cpp

    src/core/dtype.cpp
    src/core/device.cpp
    src/core/shape.cpp
    src/core/stride.cpp
    src/core/storage.cpp

    src/tensor/tensor.cpp
    src/tensor/manipulation.cpp

    src/autograd.cpp

    src/math/operations.cpp

    src/nn/linear.cpp
    src/nn/activation.cpp
    src/nn/sequential.cpp
    src/nn/mse_loss.cpp
        src/nn/embedding.cpp
        src/nn/positional_encoding.cpp
        src/nn/layer_norm.cpp
    src/nn/transformer_encoder_layer.cpp
    src/nn/transformer_decoder.cpp
    src/nn/transformer_decoder_layer.cpp
    src/nn/transformer_encoder.cpp
        src/nn/feed_forward.cpp
        src/nn/multi_head_attention.cpp
    src/nn/kv_cache.cpp
    src/nn/language_model.cpp
    src/nn/cross_entropy_loss.cpp
)

target_include_directories(venlacpu_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(
    venlacpu_core
    PUBLIC
        cxx_std_17
)

if (MSVC)

    target_compile_options(
        venlacpu_core
        PRIVATE
            /W4
    )

else()

    target_compile_options(
        venlacpu_core
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

endif()

# ============================================================
# TESTS
# ============================================================

if (VENLA_BUILD_TESTS)

    enable_testing()

    # --------------------------------------------------------
    # CORE
    # --------------------------------------------------------

    add_executable(
        venlacpu_core_test
        tests/core/test_core.cpp
    )

    target_link_libraries(
        venlacpu_core_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_core_test
        COMMAND venlacpu_core_test
    )

    # --------------------------------------------------------
    # TENSOR
    # --------------------------------------------------------

    add_executable(
        venlacpu_tensor_test
        tests/tensor/test_tensor.cpp
    )

    target_link_libraries(
        venlacpu_tensor_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_tensor_test
        COMMAND venlacpu_tensor_test
    )

    # --------------------------------------------------------
    # MATH
    # --------------------------------------------------------

    add_executable(
        venlacpu_math_test
        tests/math/test_math.cpp
    )

    target_link_libraries(
        venlacpu_math_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_math_test
        COMMAND venlacpu_math_test
    )

    # --------------------------------------------------------
    # LINEAR
    # --------------------------------------------------------

    add_executable(
        venlacpu_linear_test
        tests/nn/test_linear.cpp
    )

    target_link_libraries(
        venlacpu_linear_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_linear_test
        COMMAND venlacpu_linear_test
    )

    # --------------------------------------------------------
    # ACTIVATION
    # --------------------------------------------------------

    add_executable(
        venlacpu_activation_test
        tests/nn/test_activation.cpp
    )

    target_link_libraries(
        venlacpu_activation_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_activation_test
        COMMAND venlacpu_activation_test
    )

    # --------------------------------------------------------
    # SEQUENTIAL
    # --------------------------------------------------------

    add_executable(
        venlacpu_sequential_test
        tests/nn/test_sequential.cpp
    )

    target_link_libraries(
        venlacpu_sequential_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_sequential_test
        COMMAND venlacpu_sequential_test
    )

    # --------------------------------------------------------
    # AUTOGRAD
    # --------------------------------------------------------

    add_executable(
        venlacpu_autograd_test
        tests/autograd/test_autograd.cpp
    )

    target_link_libraries(
        venlacpu_autograd_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_autograd_test
        COMMAND venlacpu_autograd_test
    )

    # --------------------------------------------------------
    # MSE LOSS
    # --------------------------------------------------------

    add_executable(
        venlacpu_mse_loss_test
        tests/nn/test_mse_loss.cpp
    )

    target_link_libraries(
        venlacpu_mse_loss_test
        PRIVATE
            venlacpu_core
    )

    add_test(
        NAME venlacpu_mse_loss_test
        COMMAND venlacpu_mse_loss_test
    )

endif()

# ============================================================
# EXAMPLE
# ============================================================

if (VENLA_BUILD_EXAMPLES)

    add_executable(
        venlacpu_hello
        examples/hello.cpp
    )

    target_link_libraries(
        venlacpu_hello
        PRIVATE
            venlacpu_core
    )

endif()

# ============================================================
# INSTALL
# ============================================================

include(GNUInstallDirs)

install(
    TARGETS venlacpu_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# ============================================================
# TRANSPOSE AUTOGRAD TEST
# ============================================================

add_executable(
    venlacpu_transpose_autograd_test
    tests/autograd/test_transpose_autograd.cpp
)

target_link_libraries(
    venlacpu_transpose_autograd_test
    PRIVATE
    venlacpu_core
)

add_test(
    NAME venlacpu_transpose_autograd_test
    COMMAND venlacpu_transpose_autograd_test
)


# ============================================================
# TENSOR MANIPULATION
# ============================================================

add_executable(
    venlacpu_tensor_manipulation_test
    tests/tensor/test_manipulation.cpp
)

target_link_libraries(
    venlacpu_tensor_manipulation_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_tensor_manipulation_test
    COMMAND venlacpu_tensor_manipulation_test
)

# ============================================================
# OPTIMIZER
# ============================================================

target_sources(
    venlacpu_core
    PRIVATE
        src/optim/optimizer.cpp
)

add_executable(
    venlacpu_optimizer_test
    tests/optim/test_optimizer.cpp
)

target_link_libraries(
    venlacpu_optimizer_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_optimizer_test
    COMMAND venlacpu_optimizer_test
)

# ============================================================
# TOKENIZER / BPE
# ============================================================

target_sources(
    venlacpu_core
    PRIVATE
        src/tokenizer/vocabulary.cpp
        src/tokenizer/tokenizer.cpp
)

add_executable(
    venlacpu_tokenizer_test
    tests/tokenizer/test_tokenizer.cpp
)

target_link_libraries(
    venlacpu_tokenizer_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_tokenizer_test
    COMMAND venlacpu_tokenizer_test
)


# ============================================================
# EMBEDDING
# ============================================================

add_executable(
    venlacpu_embedding_test
    tests/nn/test_embedding.cpp
)

target_link_libraries(
    venlacpu_embedding_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_embedding_test
    COMMAND venlacpu_embedding_test
)

# ============================================================
# POSITIONAL ENCODING
# ============================================================

add_executable(
    venlacpu_positional_encoding_test
    tests/nn/test_positional_encoding.cpp
)

target_link_libraries(
    venlacpu_positional_encoding_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_positional_encoding_test
    COMMAND venlacpu_positional_encoding_test
)

# ============================================================
# LAYER NORMALIZATION
# ============================================================

add_executable(
    venlacpu_layer_norm_test
    tests/nn/test_layer_norm.cpp
)

target_link_libraries(
    venlacpu_layer_norm_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_layer_norm_test
    COMMAND venlacpu_layer_norm_test
)


# ============================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================

add_executable(
    venlacpu_multi_head_attention_test
    tests/nn/test_multi_head_attention.cpp
)

target_link_libraries(
    venlacpu_multi_head_attention_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_multi_head_attention_test
    COMMAND venlacpu_multi_head_attention_test
)


# ============================================================
# FEED FORWARD
# ============================================================

add_executable(
    venlacpu_feed_forward_test
    tests/nn/test_feed_forward.cpp
)

target_link_libraries(
    venlacpu_feed_forward_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_feed_forward_test
    COMMAND venlacpu_feed_forward_test
)


# ============================================================
# TRANSFORMER ENCODER LAYER
# ============================================================

add_executable(
    venlacpu_transformer_encoder_layer_test
    tests/nn/test_transformer_encoder_layer.cpp
)

target_link_libraries(
    venlacpu_transformer_encoder_layer_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_transformer_encoder_layer_test
    COMMAND venlacpu_transformer_encoder_layer_test
)


# ============================================================
# TRANSFORMER ENCODER
# ============================================================

add_executable(
    venlacpu_transformer_encoder_test
    tests/nn/test_transformer_encoder.cpp
)

target_link_libraries(
    venlacpu_transformer_encoder_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_transformer_encoder_test
    COMMAND venlacpu_transformer_encoder_test
)


# ============================================================
# TRANSFORMER DECODER LAYER
# ============================================================

add_executable(
    venlacpu_transformer_decoder_layer_test
    tests/nn/test_transformer_decoder_layer.cpp
)

target_link_libraries(
    venlacpu_transformer_decoder_layer_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_transformer_decoder_layer_test
    COMMAND venlacpu_transformer_decoder_layer_test
)


# ============================================================
# TRANSFORMER DECODER
# ============================================================

add_executable(
    venlacpu_transformer_decoder_test
    tests/nn/test_transformer_decoder.cpp
)

target_link_libraries(
    venlacpu_transformer_decoder_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_transformer_decoder_test
    COMMAND venlacpu_transformer_decoder_test
)


# ============================================================
# KV CACHE
# ============================================================

add_executable(
    venlacpu_kv_cache_test
    tests/nn/test_kv_cache.cpp
)

target_link_libraries(
    venlacpu_kv_cache_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_kv_cache_test
    COMMAND venlacpu_kv_cache_test
)

# ============================================================
# CACHED MULTI-HEAD ATTENTION
# ============================================================

add_executable(
    venlacpu_cached_multi_head_attention_test
    tests/nn/test_cached_multi_head_attention.cpp
)

target_link_libraries(
    venlacpu_cached_multi_head_attention_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_cached_multi_head_attention_test
    COMMAND venlacpu_cached_multi_head_attention_test
)

# ============================================================
# CACHED TRANSFORMER DECODER LAYER
# ============================================================

add_executable(
    venlacpu_transformer_decoder_layer_cached_test
    tests/nn/test_transformer_decoder_layer_cached.cpp
)

target_link_libraries(
    venlacpu_transformer_decoder_layer_cached_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_transformer_decoder_layer_cached_test
    COMMAND venlacpu_transformer_decoder_layer_cached_test
)

# ============================================================
# LANGUAGE MODEL
# ============================================================

add_executable(
    venlacpu_language_model_test
    tests/nn/test_language_model.cpp
)

add_executable(
    venlacpu_language_model_parameters_test
    tests/nn/test_language_model_parameters.cpp
)

target_link_libraries(
    venlacpu_language_model_parameters_test
    PRIVATE
    venlacpu_core
)

add_test(
    NAME venlacpu_language_model_parameters_test
    COMMAND venlacpu_language_model_parameters_test
)

target_link_libraries(
    venlacpu_language_model_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_language_model_test
    COMMAND venlacpu_language_model_test
)

# ============================================================
# CROSS ENTROPY LOSS
# ============================================================

add_executable(
    venlacpu_cross_entropy_loss_test
    tests/nn/test_cross_entropy_loss.cpp
)

target_link_libraries(
    venlacpu_cross_entropy_loss_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_cross_entropy_loss_test
    COMMAND venlacpu_cross_entropy_loss_test
)

add_executable(
    venlacpu_language_model_training_test
    tests/nn/test_language_model_training.cpp
)

target_link_libraries(
    venlacpu_language_model_training_test
    PRIVATE
    venlacpu_core
)

add_test(
    NAME venlacpu_language_model_training_test
    COMMAND venlacpu_language_model_training_test
)

# ============================================================
# TRAINING SYSTEM
# ============================================================

target_sources(
    venlacpu_core
    PRIVATE
        src/training/causal_lm.cpp
        src/training/trainer.cpp
)

add_executable(
    venlacpu_training_system_test
    tests/training/test_training_system.cpp
)

target_link_libraries(
    venlacpu_training_system_test
    PRIVATE
        venlacpu_core
)

add_test(
    NAME venlacpu_training_system_test
    COMMAND venlacpu_training_system_test
)
