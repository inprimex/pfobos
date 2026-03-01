"""
Build script for libfobos_stub.so.

Usage:
    uv run python tests/stub/build.py        # build
    uv run python tests/stub/build.py clean  # remove compiled .so

The compiled library is placed at tests/stub/libfobos.so so that
FobosSDR(lib_path=STUB_LIB_PATH) can load it directly.
"""

import os
import subprocess
import sys

STUB_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC       = os.path.join(STUB_DIR, "libfobos_stub.c")
OUT       = os.path.join(STUB_DIR, "libfobos.so")


def build():
    cmd = [
        "gcc",
        "-shared", "-fPIC",
        "-o", OUT,
        SRC,
        "-lm",
        "-O2",
    ]
    print(f"Building stub: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Build FAILED:")
        print(result.stderr)
        sys.exit(1)
    print(f"Built: {OUT}")


def clean():
    if os.path.exists(OUT):
        os.remove(OUT)
        print(f"Removed: {OUT}")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        build()
