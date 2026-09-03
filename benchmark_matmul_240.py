#!/usr/bin/env python3

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(ROOT, "python")

if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

import venlacpu


def make_matrix(rows, cols):
    """
    Create a Float32 matrix using the official VENLACPU API.
    """
    return venlacpu.ones((rows, cols))


def benchmark(size, warmup=2, runs=5):
    a = make_matrix(size, size)
    b = make_matrix(size, size)

    # Warmup
    for _ in range(warmup):
        c = venlacpu.matmul(a, b)

    # Timed runs
    start = time.perf_counter()

    for _ in range(runs):
        c = venlacpu.matmul(a, b)

    elapsed = time.perf_counter() - start

    avg_seconds = elapsed / runs
    avg_ms = avg_seconds * 1000.0

    # GEMM FLOPs = 2 * M * N * K
    operations = 2.0 * size * size * size
    gflops = operations / avg_seconds / 1.0e9

    # Materialize result and calculate checksum.
    values = c.tolist()

    checksum = 0.0

    for row in values:
        if isinstance(row, list):
            for value in row:
                checksum += float(value)
        else:
            checksum += float(row)

    return avg_ms, gflops, checksum


def main():
    print("=" * 72)
    print(" VENLACPU 2.4.0 — CACHE-BLOCKED GEMM BENCHMARK")
    print("=" * 72)
    print()
    print(f"Version : {venlacpu.__version__}")
    print("Warmup  : 2")
    print("Runs    : 5")
    print()

    sizes = [64, 128, 256, 512, 768]

    print(
        f"{'SIZE':>8}"
        f"{'AVG ms':>16}"
        f"{'GFLOPS':>18}"
        f"{'CHECKSUM':>20}"
    )

    print("-" * 72)

    for size in sizes:
        avg_ms, gflops, checksum = benchmark(size)

        print(
            f"{size:8d}"
            f"{avg_ms:16.3f}"
            f"{gflops:18.3f}"
            f"{checksum:20.3f}"
        )

    print("-" * 72)
    print()
    print("[PASS] GEMM benchmark completed successfully.")
    print()


if __name__ == "__main__":
    main()
