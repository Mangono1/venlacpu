#include "venla/math/simd.hpp"

#include <cstddef>

#if defined(__aarch64__) || defined(__arm__)

    #include <arm_neon.h>

    #define VENLA_SIMD_ARM_NEON 1

#else

    #define VENLA_SIMD_ARM_NEON 0

#endif


#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

    #define VENLA_SIMD_X86 1

#else

    #define VENLA_SIMD_X86 0

#endif


#if VENLA_SIMD_X86 && (defined(__GNUC__) || defined(__clang__))

    #define VENLA_HAS_X86_TARGET_ATTR 1

#else

    #define VENLA_HAS_X86_TARGET_ATTR 0

#endif


namespace venla {
namespace simd {

namespace {


#if !VENLA_SIMD_ARM_NEON

inline void accumulate_scalar(
    float* c,
    const float* b,
    float a,
    std::size_t length
) {
    for (std::size_t j = 0; j < length; ++j) {
        c[j] += a * b[j];
    }
}

#endif


#if VENLA_SIMD_ARM_NEON

void accumulate_neon(
    float* c,
    const float* b,
    float a,
    std::size_t length
) {
    const float32x4_t va = vdupq_n_f32(a);

    std::size_t j = 0;

    /*
     * Hybrid policy:
     *
     * Small/medium rows:
     *   4-wide NEON
     *
     * Large rows:
     *   8-wide unrolled NEON
     *
     * Benchmark VENLACPU Android ARM64 menunjukkan bahwa aggressive
     * 8-wide unrolling membantu beberapa ukuran tetapi merugikan
     * 256x256. Karena accumulate_row() menerima panjang baris,
     * dispatch berdasarkan panjang row dilakukan di sini.
     */

    constexpr std::size_t HYBRID_THRESHOLD = 512;

    if (length < HYBRID_THRESHOLD) {

        /*
         * Standard 4-wide NEON path.
         *
         * Dipertahankan untuk menghindari regression pada workload
         * kecil/menengah.
         */
        for (; j + 4 <= length; j += 4) {

            float32x4_t vc =
                vld1q_f32(c + j);

            const float32x4_t vb =
                vld1q_f32(b + j);

            vc = vmlaq_f32(
                vc,
                va,
                vb
            );

            vst1q_f32(
                c + j,
                vc
            );
        }

    } else {

        /*
         * Large-row 8-wide path.
         *
         * Dua accumulator independen memberi compiler lebih banyak
         * instruction-level parallelism.
         */
        for (; j + 8 <= length; j += 8) {

            float32x4_t c0 =
                vld1q_f32(c + j);

            float32x4_t c1 =
                vld1q_f32(c + j + 4);

            const float32x4_t b0 =
                vld1q_f32(b + j);

            const float32x4_t b1 =
                vld1q_f32(b + j + 4);

            c0 = vmlaq_f32(
                c0,
                va,
                b0
            );

            c1 = vmlaq_f32(
                c1,
                va,
                b1
            );

            vst1q_f32(
                c + j,
                c0
            );

            vst1q_f32(
                c + j + 4,
                c1
            );
        }

        /*
         * 4-wide remainder setelah 8-wide loop.
         */
        for (; j + 4 <= length; j += 4) {

            float32x4_t vc =
                vld1q_f32(c + j);

            const float32x4_t vb =
                vld1q_f32(b + j);

            vc = vmlaq_f32(
                vc,
                va,
                vb
            );

            vst1q_f32(
                c + j,
                vc
            );
        }
    }

    /*
     * Scalar remainder.
     */
    for (; j < length; ++j) {
        c[j] += a * b[j];
    }

}

#endif


#if VENLA_SIMD_X86 && VENLA_HAS_X86_TARGET_ATTR

__attribute__((target("sse2")))
void accumulate_sse2(
    float* c,
    const float* b,
    float a,
    std::size_t length
) {
    const __attribute__((vector_size(16))) float va = {
        a, a, a, a
    };

    std::size_t j = 0;

    for (; j + 4 <= length; j += 4) {

        __attribute__((vector_size(16))) float vc;
        __attribute__((vector_size(16))) float vb;

        vc = *reinterpret_cast<
            const __attribute__((vector_size(16))) float*
        >(c + j);

        vb = *reinterpret_cast<
            const __attribute__((vector_size(16))) float*
        >(b + j);

        vc += va * vb;

        *reinterpret_cast<
            __attribute__((vector_size(16))) float*
        >(c + j) = vc;
    }

    for (; j < length; ++j) {
        c[j] += a * b[j];
    }
}


__attribute__((target("avx2")))
void accumulate_avx2(
    float* c,
    const float* b,
    float a,
    std::size_t length
) {
    const __attribute__((vector_size(32))) float va = {
        a, a, a, a,
        a, a, a, a
    };

    std::size_t j = 0;

    for (; j + 8 <= length; j += 8) {

        __attribute__((vector_size(32))) float vc;
        __attribute__((vector_size(32))) float vb;

        vc = *reinterpret_cast<
            const __attribute__((vector_size(32))) float*
        >(c + j);

        vb = *reinterpret_cast<
            const __attribute__((vector_size(32))) float*
        >(b + j);

        vc += va * vb;

        *reinterpret_cast<
            __attribute__((vector_size(32))) float*
        >(c + j) = vc;
    }

    for (; j < length; ++j) {
        c[j] += a * b[j];
    }
}

#endif


#if VENLA_SIMD_X86 && VENLA_HAS_X86_TARGET_ATTR

Backend detect_x86_backend() {

    #if defined(__GNUC__) || defined(__clang__)

        if (__builtin_cpu_supports("avx2")) {
            return Backend::AVX2;
        }

        if (__builtin_cpu_supports("sse2")) {
            return Backend::SSE2;
        }

    #endif

    return Backend::Scalar;
}

#endif


} // namespace


Backend backend() {

    /*
     * Detect the backend only once.
     *
     * GEMM calls accumulate_row() inside the hot k-loop.
     * Repeating CPU feature detection there would add avoidable
     * dispatch overhead.
     */

    static const Backend selected_backend = []() {

#if VENLA_SIMD_ARM_NEON

        return Backend::NEON;

#elif VENLA_SIMD_X86 && VENLA_HAS_X86_TARGET_ATTR

        return detect_x86_backend();

#else

        return Backend::Scalar;

#endif

    }();

    return selected_backend;
}


const char* backend_name() {

    switch (backend()) {

        case Backend::NEON:
            return "NEON";

        case Backend::AVX2:
            return "AVX2";

        case Backend::SSE2:
            return "SSE2";

        case Backend::Scalar:
        default:
            return "Scalar";
    }
}


void accumulate_row(
    float* c,
    const float* b,
    float a,
    std::size_t length
) {
    if (length == 0) {
        return;
    }

#if VENLA_SIMD_ARM_NEON

    accumulate_neon(
        c,
        b,
        a,
        length
    );

#elif VENLA_SIMD_X86 && VENLA_HAS_X86_TARGET_ATTR

    switch (backend()) {

        case Backend::AVX2:
            accumulate_avx2(
                c,
                b,
                a,
                length
            );
            return;

        case Backend::SSE2:
            accumulate_sse2(
                c,
                b,
                a,
                length
            );
            return;

        case Backend::Scalar:
        default:
            break;
    }

    accumulate_scalar(
        c,
        b,
        a,
        length
    );

#else

    accumulate_scalar(
        c,
        b,
        a,
        length
    );

#endif
}


} // namespace simd
} // namespace venla
