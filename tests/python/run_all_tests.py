#!/usr/bin/env python3

import inspect
import pathlib
import sys
import traceback
import importlib.util


ROOT = pathlib.Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "tests" / "python"

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

if str(ROOT / "build") not in sys.path:
    sys.path.insert(0, str(ROOT / "build"))


def load_module(path):
    module_name = "venlacpu_test_" + path.stem

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load test module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def discover_tests(module):
    tests = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("test_") and inspect.isfunction(obj):
            tests.append((name, obj))

    return tests


def main():
    print()
    print("=" * 60)
    print(" VENLACPU 2.3.1 — PYTHON REGRESSION RUNNER")
    print("=" * 60)
    print()

    print("Python:", sys.version)
    print("Root:", ROOT)
    print("Test directory:", TEST_DIR)
    print()

    test_files = sorted(
        TEST_DIR.glob("test_*.py")
    )

    if not test_files:
        print("[FAIL] No Python test files found.")
        return 1

    total = 0
    passed = 0
    failed = 0

    for path in test_files:

        print()
        print("-" * 60)
        print("FILE:", path.relative_to(ROOT))
        print("-" * 60)

        try:
            module = load_module(path)
        except Exception:
            failed += 1

            print("[FAIL] Module import failed")
            traceback.print_exc()

            continue

        tests = discover_tests(module)

        if not tests:
            print("[WARN] No test_* functions found.")
            continue

        for name, test in tests:

            total += 1

            print()
            print("RUN:", name)

            try:
                test()

            except Exception as exc:
                failed += 1

                print(
                    "[FAIL]",
                    name,
                    "->",
                    type(exc).__name__,
                    ":",
                    exc,
                )

                traceback.print_exc()

            else:
                passed += 1
                print("[PASS]", name)

    print()
    print("=" * 60)
    print(" PYTHON REGRESSION SUMMARY")
    print("=" * 60)
    print()
    print("Files :", len(test_files))
    print("Tests :", total)
    print("PASS  :", passed)
    print("FAIL  :", failed)
    print()

    if failed:
        print("[FAIL] Python regression suite gagal.")
        return 1

    if total == 0:
        print("[FAIL] Tidak ada test function yang dijalankan.")
        return 1

    print("[PASS] Semua Python regression test berhasil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
