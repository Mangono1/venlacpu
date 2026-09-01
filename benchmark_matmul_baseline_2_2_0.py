#!/usr/bin/env python3

import os
import sys
import time
import statistics


ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON_DIR = os.path.join(ROOT, "python")
BUILD_DIR = os.path.join(ROOT, "build")

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, BUILD_DIR)


import venlacpu


print()
print("=" * 68)
print(" VENLACPU 2.2.0 — MATMUL BASELINE")
print("=" * 68)
print()

print("Package version :", venlacpu.__version__)
print("Native available:", venlacpu.native_available())
print("CPU cores       :", os.cpu_count())

assert venlacpu.__version__ == "2.2.0"
assert venlacpu.native_available()


# ============================================================
# VALUE EXTRACTION
# ============================================================

def first_scalar(value):
    """
    Extract the first scalar recursively from Tensor.tolist().

    VENLACPU tolist() may return:
        float
        [float, ...]
        [[float, ...], ...]
        [[[float, ...]], ...]
    """

    while isinstance(value, list):
        if not value:
            raise RuntimeError(
                "Tensor.tolist() menghasilkan list kosong"
            )

        value = value[0]

    return float(value)


def tensor_first_value(tensor):
    return first_scalar(tensor.tolist())


# ============================================================
# TENSOR CREATION
# ============================================================

def make_matrix(rows, cols):
    return venlacpu.Tensor.ones(
        venlacpu.Shape([rows, cols]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )


def make_tensor(shape):
    return venlacpu.Tensor.ones(
        venlacpu.Shape(list(shape)),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )


# ============================================================
# CORRECTNESS
# ============================================================

def correctness_test():

    print()
    print("-" * 68)
    print("CORRECTNESS")
    print("-" * 68)

    tests = [
        ("1D x 1D", (16,), (16,)),
        ("2D x 1D", (16, 32), (32,)),
        ("1D x 2D", (32,), (32, 16)),
        ("2D x 2D", (32, 64), (64, 16)),
        ("Batched 3D x 3D", (2, 32, 64), (2, 64, 16)),
        ("Broadcast batch", (1, 32, 64), (4, 64, 16)),
    ]

    for name, ashape, bshape in tests:

        a = make_tensor(ashape)
        b = make_tensor(bshape)

        result = venlacpu.matmul(a, b)

        dims = result.shape().dimensions()

        if len(ashape) == 1 and len(bshape) == 1:
            expected = float(ashape[0])
        else:
            expected = float(ashape[-1])

        actual = tensor_first_value(result)

        assert actual == expected, (
            f"{name}: expected {expected}, got {actual}"
        )

        print(
            f"[PASS] {name:<20} "
            f"shape={dims} "
            f"value={actual}"
        )

    print()
    print("[PASS] All matmul correctness tests")


# ============================================================
# SINGLE BENCHMARK CASE
# ============================================================

def benchmark_case(
    n,
    warmup=2,
    runs=7,
):

    a = make_matrix(n, n)
    b = make_matrix(n, n)

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup):
        result = venlacpu.matmul(a, b)

        value = tensor_first_value(result)

        assert value == float(n)

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    samples = []

    for _ in range(runs):

        start = time.perf_counter()

        result = venlacpu.matmul(a, b)

        elapsed = time.perf_counter() - start

        samples.append(elapsed)

        value = tensor_first_value(result)

        assert value == float(n)

    median = statistics.median(samples)
    best = min(samples)

    operations = 2.0 * n * n * n

    gflops = (
        operations /
        median /
        1_000_000_000.0
    )

    return median, best, gflops


# ============================================================
# PERFORMANCE
# ============================================================

def benchmark():

    print()
    print("-" * 68)
    print("PERFORMANCE BASELINE")
    print("-" * 68)

    print()
    print(
        f"{'N':>6} "
        f"{'Median(s)':>14} "
        f"{'Best(s)':>14} "
        f"{'GFLOP/s':>14}"
    )

    print("-" * 68)

    for n in (
        64,
        128,
        256,
        384,
    ):

        median, best, gflops = benchmark_case(n)

        print(
            f"{n:>6} "
            f"{median:>14.6f} "
            f"{best:>14.6f} "
            f"{gflops:>14.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    correctness_test()

    benchmark()

    print()
    print("=" * 68)
    print(" BASELINE COMPLETE")
    print("=" * 68)
    print()

    print("Reference:")
    print("  VENLACPU 2.2.0")
    print("  Branch: optimization/cpu-matmul")
    print("  Kernel: current row-column-k")
    print()

    print("No VENLACPU source files were modified by this benchmark.")
    print()


if __name__ == "__main__":
    main()
