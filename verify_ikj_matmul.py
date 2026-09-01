#!/usr/bin/env python3

import os
import sys
import time
import statistics

ROOT = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "build"))

import venlacpu


def flatten_list(value):
    if not isinstance(value, list):
        return [value]

    result = []

    for item in value:
        result.extend(flatten_list(item))

    return result


def scalar_at(tensor, index):
    """
    Mengambil satu nilai dari tensor secara robust.

    Binding VENLACPU saat ini dapat mengembalikan tolist()
    dalam beberapa bentuk:

        scalar tensor -> [value]
        vector        -> [v0, v1, ...]
        matrix        -> nested list atau flat list

    Karena itu kita menggunakan shape untuk menghitung
    flat offset dan kemudian mengambil elemen dari hasil
    flatten.
    """

    shape = list(tensor.shape().dimensions())
    data = tensor.tolist()

    flat = flatten_list(data)

    if len(shape) == 0:
        if len(flat) != 1:
            raise AssertionError(
                "Scalar tensor menghasilkan jumlah elemen "
                f"tidak valid: {len(flat)}"
            )

        return float(flat[0])

    if len(index) != len(shape):
        raise IndexError(
            f"Index {index} tidak cocok dengan shape {shape}"
        )

    offset = 0
    stride = 1

    for dimension in reversed(range(len(shape))):
        coordinate = index[dimension]
        size = shape[dimension]

        if coordinate < 0 or coordinate >= size:
            raise IndexError(
                f"Index {index} di luar shape {shape}"
            )

        offset += coordinate * stride
        stride *= size

    if offset >= len(flat):
        raise AssertionError(
            f"Flat offset {offset} di luar data "
            f"dengan {len(flat)} elemen"
        )

    return float(flat[offset])


def make_tensor(shape):
    return venlacpu.Tensor.ones(
        venlacpu.Shape(shape),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )


def check(
    name,
    ashape,
    bshape,
    expected_shape,
    expected_value,
):
    a = make_tensor(ashape)
    b = make_tensor(bshape)

    result = venlacpu.matmul(a, b)

    actual_shape = list(
        result.shape().dimensions()
    )

    assert actual_shape == expected_shape, (
        f"{name}: shape salah. "
        f"expected={expected_shape}, "
        f"actual={actual_shape}"
    )

    if len(actual_shape) == 0:
        index = []
    else:
        index = [
            0
            for _ in actual_shape
        ]

    actual = scalar_at(
        result,
        index
    )

    assert actual == expected_value, (
        f"{name}: value salah. "
        f"expected={expected_value}, "
        f"actual={actual}"
    )

    print(
        f"[PASS] {name:<20} "
        f"shape={actual_shape} "
        f"value={actual}"
    )


def correctness():
    print()
    print("=" * 68)
    print(" VENLACPU 2.2.1 — I-K-J MATMUL CORRECTNESS")
    print("=" * 68)

    print()
    print("Package :", venlacpu.__version__)
    print("Native  :", venlacpu.native_available())
    print("CPU     :", os.cpu_count())

    assert venlacpu.__version__ == "2.2.1"
    assert venlacpu.native_available()

    print()
    print("-" * 68)
    print("SHAPE TESTS")
    print("-" * 68)

    check(
        "1D x 1D",
        [16],
        [16],
        [],
        16.0,
    )

    check(
        "2D x 1D",
        [16, 32],
        [32],
        [16],
        32.0,
    )

    check(
        "1D x 2D",
        [32],
        [32, 16],
        [16],
        32.0,
    )

    check(
        "2D x 2D",
        [32, 64],
        [64, 16],
        [32, 16],
        64.0,
    )

    check(
        "3D x 3D",
        [2, 32, 64],
        [2, 64, 16],
        [2, 32, 16],
        64.0,
    )

    check(
        "Broadcast batch",
        [1, 32, 64],
        [4, 64, 16],
        [4, 32, 16],
        64.0,
    )

    print()
    print("[PASS] All I-K-J matmul correctness tests")


def benchmark_case(
    n,
    warmup=3,
    runs=7,
):
    a = make_tensor([n, n])
    b = make_tensor([n, n])

    for _ in range(warmup):
        result = venlacpu.matmul(a, b)

        value = scalar_at(
            result,
            [0, 0]
        )

        assert value == float(n)

    samples = []

    for _ in range(runs):
        start = time.perf_counter()

        result = venlacpu.matmul(a, b)

        elapsed = (
            time.perf_counter() -
            start
        )

        value = scalar_at(
            result,
            [0, 0]
        )

        assert value == float(n)

        samples.append(elapsed)

    median = statistics.median(samples)
    best = min(samples)

    operations = (
        2.0 *
        n *
        n *
        n
    )

    gflops = (
        operations /
        median /
        1_000_000_000.0
    )

    return (
        median,
        best,
        gflops,
    )


def benchmark():
    print()
    print("=" * 68)
    print(" I-K-J PERFORMANCE")
    print("=" * 68)

    print()
    print(
        f"{'N':>6} "
        f"{'Median(s)':>14} "
        f"{'Best(s)':>14} "
        f"{'GFLOP/s':>14}"
    )

    print("-" * 68)

    results = {}

    for n in (
        64,
        128,
        256,
        384,
    ):
        median, best, gflops = (
            benchmark_case(n)
        )

        results[n] = (
            median,
            best,
            gflops,
        )

        print(
            f"{n:>6} "
            f"{median:>14.6f} "
            f"{best:>14.6f} "
            f"{gflops:>14.4f}"
        )

    return results


def main():
    correctness()

    results = benchmark()

    print()
    print("=" * 68)
    print(" I-K-J VERIFICATION COMPLETE")
    print("=" * 68)

    print()
    print("Source:")
    print("  src/math/operations.cpp")

    print()
    print("Kernel:")
    print("  row -> k -> column")

    print()
    print("Version:")
    print("  2.2.1 release API")

    print()
    print("Benchmark results:")

    for n, (
        median,
        best,
        gflops,
    ) in results.items():
        print(
            f"  N={n:<4} "
            f"median={median:.6f}s "
            f"best={best:.6f}s "
            f"GFLOP/s={gflops:.4f}"
        )

    print()
    print("No version bump.")
    print("No commit.")
    print("No tag.")
    print("No push.")
    print()


if __name__ == "__main__":
    main()
