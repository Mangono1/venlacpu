#!/usr/bin/env python3

from pathlib import Path
import sys


CMAKE_FILE = Path("CMakeLists.txt")

OLD_BLOCK = """    target_compile_features(
        _venlacpu
        PRIVATE
            cxx_std_17
    )

endif()
"""

NEW_BLOCK = """    target_compile_features(
        _venlacpu
        PRIVATE
            cxx_std_17
    )

    # ========================================================
    # PYTHON WHEEL INSTALL
    #
    # The _venlacpu extension is a CMake MODULE target.
    # It must be installed into the Python package directory
    # so scikit-build-core includes it in the wheel.
    #
    # Without this rule the extension is successfully built
    # but is omitted from the wheel.
    # ========================================================

    install(
        TARGETS _venlacpu
        LIBRARY DESTINATION venlacpu
        RUNTIME DESTINATION venlacpu
        ARCHIVE DESTINATION venlacpu
    )

endif()
"""


def main() -> int:
    print("=" * 64)
    print("VENLACPU - FIX PYTHON WHEEL PACKAGING")
    print("=" * 64)

    if not CMAKE_FILE.exists():
        print()
        print("ERROR: CMakeLists.txt tidak ditemukan.")
        print(f"Current directory: {Path.cwd()}")
        print()
        print("Jalankan script ini dari root repository venlacpu.")
        return 1

    text = CMAKE_FILE.read_text(encoding="utf-8")

    # Already fixed?
    if "TARGETS _venlacpu" in text:
        print()
        print("[OK] CMakeLists.txt sudah memiliki install rule _venlacpu.")
        print("[OK] Tidak ada perubahan yang diperlukan.")
        return 0

    # Make sure we are patching the exact Python binding block.
    count = text.count(OLD_BLOCK)

    if count != 1:
        print()
        print("[ERROR] Anchor Python binding tidak ditemukan secara unik.")
        print(f"[ERROR] Jumlah anchor ditemukan: {count}")
        print()
        print("Saya sengaja menghentikan proses agar CMakeLists.txt")
        print("tidak rusak karena patch diterapkan pada lokasi yang salah.")
        return 2

    # Backup first.
    backup = Path("CMakeLists.txt.before-wheel-fix")

    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print()
        print(f"[BACKUP] {backup}")

    patched = text.replace(OLD_BLOCK, NEW_BLOCK, 1)

    CMAKE_FILE.write_text(patched, encoding="utf-8")

    # Verify.
    verify = CMAKE_FILE.read_text(encoding="utf-8")

    required_lines = [
        "install(",
        "TARGETS _venlacpu",
        "LIBRARY DESTINATION venlacpu",
        "RUNTIME DESTINATION venlacpu",
        "ARCHIVE DESTINATION venlacpu",
    ]

    missing = [
        line
        for line in required_lines
        if line not in verify
    ]

    if missing:
        print()
        print("[ERROR] Patch gagal diverifikasi.")
        print("Missing:")
        for item in missing:
            print(f"  - {item}")
        return 3

    print()
    print("[SUCCESS] CMakeLists.txt berhasil diperbaiki.")
    print()
    print("Install rule sekarang:")
    print()
    print("    install(")
    print("        TARGETS _venlacpu")
    print("        LIBRARY DESTINATION venlacpu")
    print("        RUNTIME DESTINATION venlacpu")
    print("        ARCHIVE DESTINATION venlacpu")
    print("    )")
    print()
    print("Artinya:")
    print("  Linux   -> _venlacpu*.so masuk venlacpu/")
    print("  macOS   -> _venlacpu*.so masuk venlacpu/")
    print("  Windows -> _venlacpu*.pyd masuk venlacpu/")
    print()
    print("[NEXT] Build wheel lagi.")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
