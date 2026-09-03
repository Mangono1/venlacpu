#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

#include "venla/math/operations.hpp"
#include "venla/math/simd.hpp"

namespace {

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    assert(
        std::fabs(actual - expected) <=
        tolerance
    );
}

} // namespace


// ============================================================
// SIMD KERNEL REGRESSION TEST
// ============================================================

static void test_simd_kernel() {

    constexpr std::size_t SIZE = 37;

    /*
     * 37 deliberately exercises:
     *
     *   32 elements -> SIMD vector path
     *    5 elements -> scalar remainder
     *
     * This verifies that the dispatcher handles
     * both the vectorized block and the tail correctly.
     */

    std::vector<float> c(
        SIZE,
        1.0f
    );

    std::vector<float> b(
        SIZE
    );

    for (std::size_t i = 0;
         i < SIZE;
         ++i) {

        b[i] =
            static_cast<float>(
                (i % 7) + 1
            );
    }

    venla::simd::accumulate_row(
        c.data(),
        b.data(),
        2.0f,
        SIZE
    );

    for (std::size_t i = 0;
         i < SIZE;
         ++i) {

        const float expected =
            1.0f +
            2.0f * b[i];

        assert(
            std::abs(
                c[i] - expected
            ) < 1e-5f
        );
    }

    /*
     * Also test zero-length input.
     */

    venla::simd::accumulate_row(
        c.data(),
        b.data(),
        5.0f,
        0
    );

    std::cout
        << "[OK] SIMD backend: "
        << venla::simd::backend_name()
        << std::endl;
}

int main() {

    test_simd_kernel();


    // ========================================================
    // ELEMENTWISE
    // ========================================================

    {
        auto a =
            venla::Tensor::zeros(
                {2, 3}
            );

        auto b =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* a_data =
            a.data_as<float>();

        float* b_data =
            b.data_as<float>();

        for (std::size_t i = 0;
             i < 6;
             ++i) {

            a_data[i] =
                static_cast<float>(i + 1);

            b_data[i] =
                static_cast<float>((i + 1) * 10);
        }

        auto c =
            venla::add(a, b);

        assert(c.shape().to_string() ==
               "[2, 3]");

        assert_close(
            c.data_as<float>()[0],
            11.0f
        );

        assert_close(
            c.data_as<float>()[5],
            66.0f
        );

        auto d =
            venla::sub(b, a);

        assert_close(
            d.data_as<float>()[0],
            9.0f
        );

        auto e =
            venla::mul(a, b);

        assert_close(
            e.data_as<float>()[2],
            90.0f
        );

        auto f =
            venla::div(b, a);

        assert_close(
            f.data_as<float>()[5],
            10.0f
        );
    }

    // ========================================================
    // BROADCAST
    // ========================================================

    {
        auto a =
            venla::Tensor::zeros(
                {2, 3}
            );

        auto b =
            venla::Tensor::ones(
                {3}
            );

        float* a_data =
            a.data_as<float>();

        for (std::size_t i = 0;
             i < 6;
             ++i) {

            a_data[i] =
                static_cast<float>(i + 1);
        }

        auto c =
            venla::add(a, b);

        assert(c.shape().to_string() ==
               "[2, 3]");

        assert_close(
            c.data_as<float>()[0],
            2.0f
        );

        assert_close(
            c.data_as<float>()[1],
            3.0f
        );

        assert_close(
            c.data_as<float>()[2],
            4.0f
        );

        assert_close(
            c.data_as<float>()[5],
            7.0f
        );
    }

    // ========================================================
    // NEG
    // ========================================================

    {
        auto a =
            venla::Tensor::ones(
                {4}
            );

        auto b =
            venla::neg(a);

        for (std::size_t i = 0;
             i < 4;
             ++i) {

            assert_close(
                b.data_as<float>()[i],
                -1.0f
            );
        }
    }

    // ========================================================
    // REDUCTION
    // ========================================================

    {
        auto x =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* data =
            x.data_as<float>();

        data[0] = 1.0f;
        data[1] = 2.0f;
        data[2] = 3.0f;
        data[3] = 4.0f;
        data[4] = 5.0f;
        data[5] = 6.0f;

        auto s =
            venla::sum(x);

        assert(s.ndim() == 0);
        assert_close(
            s.data_as<float>()[0],
            21.0f
        );

        auto m =
            venla::mean(x);

        assert_close(
            m.data_as<float>()[0],
            3.5f
        );

        auto maximum =
            venla::max(x);

        assert_close(
            maximum.data_as<float>()[0],
            6.0f
        );

        auto minimum =
            venla::min(x);

        assert_close(
            minimum.data_as<float>()[0],
            1.0f
        );
    }

    // ========================================================
    // DOT PRODUCT
    // ========================================================

    {
        auto a =
            venla::Tensor::zeros(
                {3}
            );

        auto b =
            venla::Tensor::zeros(
                {3}
            );

        float* a_data =
            a.data_as<float>();

        float* b_data =
            b.data_as<float>();

        a_data[0] = 1.0f;
        a_data[1] = 2.0f;
        a_data[2] = 3.0f;

        b_data[0] = 4.0f;
        b_data[1] = 5.0f;
        b_data[2] = 6.0f;

        auto result =
            venla::dot(a, b);

        assert(result.ndim() == 0);

        assert_close(
            result.data_as<float>()[0],
            32.0f
        );
    }

    // ========================================================
    // 2D MATRIX MULTIPLICATION
    //
    // [2,3] @ [3,2]
    //
    // [1 2 3]   [ 7  8]
    // [4 5 6] x [ 9 10]
    //           [11 12]
    //
    // = [ 58  64]
    //   [139 154]
    // ========================================================

    {
        auto A =
            venla::Tensor::zeros(
                {2, 3}
            );

        auto B =
            venla::Tensor::zeros(
                {3, 2}
            );

        float* a =
            A.data_as<float>();

        float* b =
            B.data_as<float>();

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;

        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;

        b[2] = 9;
        b[3] = 10;

        b[4] = 11;
        b[5] = 12;

        auto C =
            venla::matmul(A, B);

        assert(C.shape().to_string() ==
               "[2, 2]");

        float* c =
            C.data_as<float>();

        assert_close(c[0], 58);
        assert_close(c[1], 64);
        assert_close(c[2], 139);
        assert_close(c[3], 154);
    }

    // ========================================================
    // 2D x 1D
    // ========================================================

    {
        auto A =
            venla::Tensor::zeros(
                {2, 3}
            );

        auto x =
            venla::Tensor::zeros(
                {3}
            );

        float* a =
            A.data_as<float>();

        float* v =
            x.data_as<float>();

        a[0] = 1;
        a[1] = 2;
        a[2] = 3;

        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        v[0] = 10;
        v[1] = 20;
        v[2] = 30;

        auto result =
            venla::matmul(A, x);

        assert(result.shape().to_string() ==
               "[2]");

        assert_close(
            result.data_as<float>()[0],
            140
        );

        assert_close(
            result.data_as<float>()[1],
            320
        );
    }

    // ========================================================
    // 1D x 2D
    // ========================================================

    {
        auto x =
            venla::Tensor::zeros(
                {3}
            );

        auto B =
            venla::Tensor::zeros(
                {3, 2}
            );

        float* v =
            x.data_as<float>();

        float* b =
            B.data_as<float>();

        v[0] = 1;
        v[1] = 2;
        v[2] = 3;

        b[0] = 7;
        b[1] = 8;

        b[2] = 9;
        b[3] = 10;

        b[4] = 11;
        b[5] = 12;

        auto result =
            venla::matmul(x, B);

        assert(result.shape().to_string() ==
               "[2]");

        assert_close(
            result.data_as<float>()[0],
            58
        );

        assert_close(
            result.data_as<float>()[1],
            64
        );
    }

    // ========================================================
    // 1D x 1D
    // ========================================================

    {
        auto a =
            venla::Tensor::zeros(
                {3}
            );

        auto b =
            venla::Tensor::zeros(
                {3}
            );

        float* av =
            a.data_as<float>();

        float* bv =
            b.data_as<float>();

        av[0] = 1;
        av[1] = 2;
        av[2] = 3;

        bv[0] = 4;
        bv[1] = 5;
        bv[2] = 6;

        auto result =
            venla::matmul(a, b);

        assert(result.ndim() == 0);

        assert_close(
            result.data_as<float>()[0],
            32
        );
    }

    // ========================================================
    // TRANSPOSE 2D
    // ========================================================

    {
        auto A =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* a =
            A.data_as<float>();

        for (std::size_t i = 0;
             i < 6;
             ++i) {

            a[i] =
                static_cast<float>(i + 1);
        }

        auto T =
            venla::transpose(A);

        assert(T.shape().to_string() ==
               "[3, 2]");

        float* t =
            T.data_as<float>();

        assert_close(t[0], 1);
        assert_close(t[1], 4);
        assert_close(t[2], 2);
        assert_close(t[3], 5);
        assert_close(t[4], 3);
        assert_close(t[5], 6);
    }

    // ========================================================
    // 3D BATCH MATMUL
    //
    // [2,2,3] @ [2,3,2]
    //       ↓
    // [2,2,2]
    // ========================================================

    {
        auto A =
            venla::Tensor::zeros(
                {2, 2, 3}
            );

        auto B =
            venla::Tensor::zeros(
                {2, 3, 2}
            );

        float* a =
            A.data_as<float>();

        float* b =
            B.data_as<float>();

        // Batch 0
        a[0] = 1;
        a[1] = 2;
        a[2] = 3;

        a[3] = 4;
        a[4] = 5;
        a[5] = 6;

        b[0] = 7;
        b[1] = 8;

        b[2] = 9;
        b[3] = 10;

        b[4] = 11;
        b[5] = 12;

        // Batch 1
        a[6] = 1;
        a[7] = 2;
        a[8] = 3;

        a[9] = 4;
        a[10] = 5;
        a[11] = 6;

        b[6] = 7;
        b[7] = 8;

        b[8] = 9;
        b[9] = 10;

        b[10] = 11;
        b[11] = 12;

        auto C =
            venla::matmul(A, B);

        assert(C.shape().to_string() ==
               "[2, 2, 2]");

        float* c =
            C.data_as<float>();

        assert_close(c[0], 58);
        assert_close(c[1], 64);
        assert_close(c[2], 139);
        assert_close(c[3], 154);

        assert_close(c[4], 58);
        assert_close(c[5], 64);
        assert_close(c[6], 139);
        assert_close(c[7], 154);
    }

    // ========================================================
    // 4D BATCH MATMUL
    //
    // [2,3,2,3] @ [2,3,3,2]
    //              ↓
    // [2,3,2,2]
    // ========================================================

    {
        auto A =
            venla::Tensor::zeros(
                {2, 3, 2, 3}
            );

        auto B =
            venla::Tensor::zeros(
                {2, 3, 3, 2}
            );

        float* a =
            A.data_as<float>();

        float* b =
            B.data_as<float>();

        for (std::size_t i = 0;
             i < A.numel();
             ++i) {

            a[i] = 1.0f;
        }

        for (std::size_t i = 0;
             i < B.numel();
             ++i) {

            b[i] = 2.0f;
        }

        auto C =
            venla::matmul(A, B);

        assert(C.shape().to_string() ==
               "[2, 3, 2, 2]");

        float* c =
            C.data_as<float>();

        for (std::size_t i = 0;
             i < C.numel();
             ++i) {

            assert_close(
                c[i],
                6.0f
            );
        }
    }

    // ========================================================
    // BROADCASTED BATCH MATMUL
    //
    // A = [1,2,3]
    // B = [4,3,2]
    //
    // Output = [4,2]
    //
    // A is broadcast across batch dimension.
    // ========================================================

    {
        auto A =
            venla::Tensor::ones(
                {1, 2, 3}
            );

        auto B =
            venla::Tensor::ones(
                {4, 3, 2}
            );

        auto C =
            venla::matmul(A, B);

        assert(C.shape().to_string() ==
               "[4, 2, 2]");

        float* c =
            C.data_as<float>();

        for (std::size_t i = 0;
             i < C.numel();
             ++i) {

            assert_close(
                c[i],
                3.0f
            );
        }
    }

    std::cout
        << "VENLACPU mathematics tests passed\n";

    return 0;
}
