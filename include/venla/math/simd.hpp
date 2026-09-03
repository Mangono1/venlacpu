#pragma once

#include <cstddef>

namespace venla {
namespace simd {

/*
 * VENLACPU Portable SIMD Engine
 *
 * The public kernel contract is deliberately small:
 *
 *     c[j] += a * b[j]
 *
 * for j in [0, length).
 *
 * The implementation selected internally may use:
 *
 *   - ARM NEON
 *   - x86 AVX2
 *   - x86 SSE2
 *   - scalar fallback
 *
 * Callers do not need to know which ISA is active.
 */

enum class Backend {
    Scalar,
    NEON,
    SSE2,
    AVX2
};


/*
 * Returns the backend selected for the current CPU/build.
 */
Backend backend();


/*
 * Human-readable backend name.
 */
const char* backend_name();


/*
 * Perform:

 *     c[j] += a * b[j]
 *
 * for length elements.
 *
 * c and b must point to contiguous float32 storage.
 */
void accumulate_row(
    float* c,
    const float* b,
    float a,
    std::size_t length
);


} // namespace simd
} // namespace venla
