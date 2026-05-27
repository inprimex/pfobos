#!/usr/bin/env bash
# Cross-compile libfobos + SoapyFobosSDR for aarch64 on an x86-64 host (no Docker/QEMU).
# Alternative to build-docker.sh — faster but requires multiarch apt setup.
#
# Prerequisites (Debian/Ubuntu host):
#   sudo dpkg --add-architecture arm64
#   sudo apt-get update
#   sudo apt-get install -y \
#       gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
#       libusb-1.0-0-dev:arm64 \
#       libsoapysdr-dev:arm64
#
# Usage:
#   ./setup/aarch64/build-cross.sh            # build to dist/fobos-aarch64/
#   ./setup/aarch64/build-cross.sh --clean    # wipe build dirs first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/aarch64"
STAGING="${BUILD_DIR}/staging"
OUTPUT_DIR="${REPO_ROOT}/dist/fobos-aarch64"
TOOLCHAIN="${SCRIPT_DIR}/CMakeToolchain-aarch64.cmake"

if [[ "${1:-}" == "--clean" ]]; then
    echo "--- cleaning build dirs ---"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}" "${STAGING}" "${OUTPUT_DIR}/lib" "${OUTPUT_DIR}/include" "${OUTPUT_DIR}/udev"

# Verify cross-compiler
if ! command -v aarch64-linux-gnu-gcc &>/dev/null; then
    echo "ERROR: aarch64-linux-gnu-gcc not found."
    echo "  sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
    exit 1
fi

CMAKE_COMMON=(
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${STAGING}"
)

# Helper: clone if not already present
clone_or_update() {
    local url="$1" dest="$2"
    if [[ -d "${dest}/.git" ]]; then
        echo "--- using existing clone: ${dest} ---"
    else
        git clone --depth 1 "${url}" "${dest}"
    fi
}

# libfobos
echo "=== building libfobos ==="
clone_or_update https://github.com/rigexpert/libfobos.git "${BUILD_DIR}/libfobos"
cmake -S "${BUILD_DIR}/libfobos" -B "${BUILD_DIR}/libfobos/build" "${CMAKE_COMMON[@]}"
cmake --build "${BUILD_DIR}/libfobos/build" -j"$(nproc)"
cmake --install "${BUILD_DIR}/libfobos/build"

# libfobos-sdr-agile
echo "=== building libfobos-sdr-agile ==="
clone_or_update https://github.com/rigexpert/libfobos-sdr-agile.git "${BUILD_DIR}/libfobos-sdr-agile"
cmake -S "${BUILD_DIR}/libfobos-sdr-agile" -B "${BUILD_DIR}/libfobos-sdr-agile/build" "${CMAKE_COMMON[@]}"
cmake --build "${BUILD_DIR}/libfobos-sdr-agile/build" -j"$(nproc)"
cmake --install "${BUILD_DIR}/libfobos-sdr-agile/build"

# SoapyFobosSDR — needs SoapySDR arm64 dev headers + our cross-compiled fobos libs
echo "=== building SoapyFobosSDR ==="
clone_or_update https://github.com/rigexpert/SoapyFobosSDR.git "${BUILD_DIR}/SoapyFobosSDR"

PKG_CONFIG_PATH="${STAGING}/lib/pkgconfig:${STAGING}/lib/aarch64-linux-gnu/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig" \
cmake -S "${BUILD_DIR}/SoapyFobosSDR" -B "${BUILD_DIR}/SoapyFobosSDR/build" \
    "${CMAKE_COMMON[@]}" \
    -DCMAKE_PREFIX_PATH="${STAGING}"
cmake --build "${BUILD_DIR}/SoapyFobosSDR/build" -j"$(nproc)"
cmake --install "${BUILD_DIR}/SoapyFobosSDR/build"

# Collect artifact
echo "=== collecting artifact ==="
cp -P "${STAGING}"/lib/libfobos.so*     "${OUTPUT_DIR}/lib/"
cp -P "${STAGING}"/lib/libfobos_sdr.so* "${OUTPUT_DIR}/lib/"

# SoapySDR module
find "${STAGING}/lib" -name "FobosSDRSupport.so" 2>/dev/null | head -1 \
    | xargs -I{} sh -c \
        'dir=$(dirname {}); rel=${dir#'"${STAGING}"'/lib/}; mkdir -p '"${OUTPUT_DIR}"'/lib/${rel}; cp {} '"${OUTPUT_DIR}"'/lib/${rel}/'

cp "${STAGING}/include/fobos.h"     "${OUTPUT_DIR}/include/"
cp "${STAGING}/include/fobos_sdr.h" "${OUTPUT_DIR}/include/"
cp "${BUILD_DIR}/libfobos/fobos-sdr.rules" "${OUTPUT_DIR}/udev/"

# Version manifest
{
    printf "libfobos:       %s\n" "$(git -C "${BUILD_DIR}/libfobos" rev-parse --short HEAD)"
    printf "libfobos-agile: %s\n" "$(git -C "${BUILD_DIR}/libfobos-sdr-agile" rev-parse --short HEAD)"
    printf "SoapyFobosSDR:  %s\n" "$(git -C "${BUILD_DIR}/SoapyFobosSDR" rev-parse --short HEAD)"
    printf "arch:           linux/arm64 (cross-compiled)\n"
    printf "host:           %s\n" "$(uname -m)"
    printf "toolchain:      %s\n" "$(aarch64-linux-gnu-gcc --version | head -1)"
} > "${OUTPUT_DIR}/VERSIONS"

echo ""
echo "=== done ==="
echo "artifact: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f | sort
echo ""
cat "${OUTPUT_DIR}/VERSIONS"
