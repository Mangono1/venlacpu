#!/data/data/com.termux/files/usr/bin/bash

set -e

ROOT="$HOME/venlacpu"

cd "$ROOT"

echo "============================================================"
echo " VENLACPU - AUTOGRAD TRANSPOSE FIX"
echo "============================================================"

if [ ! -f "include/venla/autograd/ops.hpp" ]; then
    echo "ERROR: include/venla/autograd/ops.hpp tidak ditemukan"
    exit 1
fi

if [ ! -f "src/autograd.cpp" ]; then
    echo "ERROR: src/autograd.cpp tidak ditemukan"
    exit 1
fi

if [ ! -f "src/math/operations.cpp" ]; then
    echo "ERROR: src/math/operations.cpp tidak ditemukan"
    exit 1
fi

if [ ! -f "include/venla/math/operations.hpp" ]; then
    echo "ERROR: include/venla/math/operations.hpp tidak ditemukan"
    exit 1
fi

if [ ! -f "include/venla/autograd/autograd.hpp" ]; then
    echo "ERROR: include/venla/autograd/autograd.hpp tidak ditemukan"
    exit 1
fi

echo
echo "============================================================"
echo " BACKUP"
echo "============================================================"

BACKUP_DIR="backup_autograd_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

cp include/venla/autograd/ops.hpp \
   "$BACKUP_DIR/ops.hpp"

cp src/autograd.cpp \
   "$BACKUP_DIR/autograd.cpp"

cp src/math/operations.cpp \
   "$BACKUP_DIR/operations.cpp"

echo "Backup dibuat di:"
echo "  $BACKUP_DIR"

echo
echo "============================================================"
echo " 1. UPDATE ops.hpp"
echo "============================================================"

python - <<'PY'
from pathlib import Path

path = Path("include/venla/autograd/ops.hpp")
text = path.read_text()

marker = """std::shared_ptr<AutogradNode>
make_matmul_node(
    const Tensor& a,
    const Tensor& b
);
"""

if marker not in text:
    raise SystemExit(
        "ERROR: deklarasi make_matmul_node tidak ditemukan"
    )

addition = """
// ============================================================
// TRANSPOSE AUTOGRAD
// ============================================================

std::shared_ptr<AutogradNode>
make_transpose_node(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
);
"""

if "make_transpose_node(" not in text:
    text = text.replace(
        marker,
        marker + addition
    )

path.write_text(text)
print("ops.hpp berhasil diperbarui.")
PY

echo
echo "============================================================"
echo " 2. UPDATE src/autograd.cpp"
echo "============================================================"

cat >> src/autograd.cpp <<'CPP'

/*
 * ============================================================
 * TRANSPOSE AUTOGRAD
 * ============================================================
 *
 * Forward:
 *
 *     y = transpose(x, dim0, dim1)
 *
 * Backward:
 *
 *     dx = transpose(dy, dim0, dim1)
 *
 * Karena transpose adalah operasi permutasi dimensi,
 * inverse permutation-nya adalah transpose dengan pasangan
 * dimensi yang sama.
 *
 * Contoh:
 *
 *     x : [2, 3]
 *     y : [3, 2]
 *
 *     dy : [3, 2]
 *     dx : [2, 3]
 *
 * Untuk dimensi yang berbeda:
 *
 *     transpose(x, 0, 2)
 *
 * gradient dikembalikan dengan:
 *
 *     transpose(gradient, 0, 2)
 *
 * ============================================================
 */

namespace venla {

std::shared_ptr<AutogradNode>
make_transpose_node(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
) {
    if (!input.requires_grad()) {
        return nullptr;
    }

    if (dim0 == dim1) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{input},

        [input, dim0, dim1](
            const Tensor& gradient
        ) mutable {

            if (gradient.shape() !=
                input.shape()) {

                // Gradient dari transpose memiliki
                // shape hasil transpose.
                //
                // Jadi validasi dilakukan berdasarkan
                // expected output shape.

                std::vector<std::size_t>
                    expected_dimensions =
                        input.shape().dimensions();

                std::swap(
                    expected_dimensions[dim0],
                    expected_dimensions[dim1]
                );

                Shape expected_shape(
                    expected_dimensions
                );

                if (gradient.shape() !=
                    expected_shape) {

                    throw std::runtime_error(
                        "transpose backward: "
                        "gradient shape mismatch"
                    );
                }
            }

            Tensor grad_input =
                transpose(
                    gradient,
                    dim0,
                    dim1
                );

            propagate(
                input,
                grad_input
            );
        }
    );
}

} // namespace venla
CPP

echo "src/autograd.cpp berhasil diperbarui."

echo
echo "============================================================"
echo " 3. UPDATE transpose() DI operations.cpp"
echo "============================================================"

python - <<'PY'
from pathlib import Path

path = Path("src/math/operations.cpp")
text = path.read_text()

old = """    for (std::size_t i = 0;
         i < result.numel();
         ++i) {

        const std::vector<std::size_t>
            output_coordinates =
                unravel_index(
                    i,
                    result.shape()
                );

        std::vector<std::size_t>
            input_coordinates(
                rank
            );

        for (std::size_t dimension = 0;
             dimension < rank;
             ++dimension) {

            if (dimension == dim0) {
                input_coordinates[dimension] =
                    output_coordinates[dim1];
            }
            else if (dimension == dim1) {
                input_coordinates[dimension] =
                    output_coordinates[dim0];
            }
            else {
                input_coordinates[dimension] =
                    output_coordinates[dimension];
            }
        }

        const std::size_t input_index =
            ravel_index(
                input_coordinates,
                input.shape()
            );

        destination[i] =
            source[input_index];
    }

    return result;
}
"""

new = """    for (std::size_t i = 0;
         i < result.numel();
         ++i) {

        const std::vector<std::size_t>
            output_coordinates =
                unravel_index(
                    i,
                    result.shape()
                );

        std::vector<std::size_t>
            input_coordinates(
                rank
            );

        for (std::size_t dimension = 0;
             dimension < rank;
             ++dimension) {

            if (dimension == dim0) {
                input_coordinates[dimension] =
                    output_coordinates[dim1];
            }
            else if (dimension == dim1) {
                input_coordinates[dimension] =
                    output_coordinates[dim0];
            }
            else {
                input_coordinates[dimension] =
                    output_coordinates[dimension];
            }
        }

        const std::size_t input_index =
            ravel_index(
                input_coordinates,
                input.shape()
            );

        destination[i] =
            source[input_index];
    }

    // --------------------------------------------------------
    // AUTOGRAD
    // --------------------------------------------------------

    if (input.requires_grad()) {

        result.set_grad_fn(
            make_transpose_node(
                input,
                dim0,
                dim1
            )
        );
    }

    return result;
}
"""

if old not in text:
    raise SystemExit(
        "ERROR: blok transpose() yang diharapkan tidak ditemukan. "
        "File tidak diubah pada tahap ini."
    )

text = text.replace(old, new, 1)

path.write_text(text)

print(
    "src/math/operations.cpp berhasil diperbarui."
)
PY

echo
echo "============================================================"
echo " 4. BUAT TEST TRANSPOSE AUTOGRAD"
echo "============================================================"

mkdir -p tests/autograd

cat > tests/autograd/test_transpose_autograd.cpp <<'CPP'
#include "venla/math/operations.hpp"
#include "venla/tensor/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

void expect_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    if (std::fabs(actual - expected) >
        tolerance) {

        throw std::runtime_error(
            "value mismatch"
        );
    }
}

void test_2d_transpose_backward() {

    venla::Tensor x =
        venla::Tensor::zeros(
            {2, 3},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    float* x_data =
        x.data_as<float>();

    x_data[0] = 1.0f;
    x_data[1] = 2.0f;
    x_data[2] = 3.0f;

    x_data[3] = 4.0f;
    x_data[4] = 5.0f;
    x_data[5] = 6.0f;

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x
        );

    if (y.shape() != venla::Shape({3, 2})) {
        throw std::runtime_error(
            "transpose produced wrong shape"
        );
    }

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "x does not have gradient"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < 6;
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

void test_3d_transpose_backward() {

    venla::Tensor x =
        venla::Tensor::ones(
            {2, 3, 4},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x,
            0,
            2
        );

    if (y.shape() !=
        venla::Shape({4, 3, 2})) {

        throw std::runtime_error(
            "3D transpose produced wrong shape"
        );
    }

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "3D transpose gradient missing"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

void test_transpose_chain() {

    venla::Tensor x =
        venla::Tensor::ones(
            {2, 3},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x
        );

    venla::Tensor z =
        venla::transpose(
            y
        );

    venla::Tensor loss =
        venla::sum(
            z
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "transpose chain gradient missing"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

} // namespace

int main() {

    try {

        test_2d_transpose_backward();
        test_3d_transpose_backward();
        test_transpose_chain();

        std::cout
            << "TRANSPOSE AUTOGRAD TEST PASSED"
            << std::endl;

        return 0;

    }
    catch (const std::exception& error) {

        std::cerr
            << "TRANSPOSE AUTOGRAD TEST FAILED: "
            << error.what()
            << std::endl;

        return 1;
    }
}
CPP

echo "Test dibuat."

echo
echo "============================================================"
echo " 5. CEK CMAKE"
echo "============================================================"

if grep -R -n \
    "venlacpu_autograd_test" \
    CMakeLists.txt cmake tests \
    --exclude-dir=build \
    2>/dev/null | grep -q .; then

    echo "Target autograd ditemukan."

else

    echo "Target test autograd belum ditemukan secara eksplisit."
    echo "Mencoba menambahkan test baru ke CMakeLists.txt."

    cat >> CMakeLists.txt <<'CMAKE'

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
CMAKE

fi

echo
echo "============================================================"
echo " 6. VERIFIKASI DECLARATION"
echo "============================================================"

grep -n -A8 -B4 \
    "make_transpose_node" \
    include/venla/autograd/ops.hpp

echo
grep -n -A12 -B4 \
    "make_transpose_node" \
    src/autograd.cpp

echo
echo "============================================================"
echo " 7. BUILD"
echo "============================================================"

rm -rf build

cmake -S . -B build

cmake --build build -j2

echo
echo "============================================================"
echo " 8. TEST"
echo "============================================================"

cd build

ctest --output-on-failure

echo
echo "============================================================"
echo " AUTOGRAD TRANSPOSE FIX SELESAI"
echo "============================================================"
