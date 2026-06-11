#!/usr/bin/env bash
# Build libfobos + libfobos-sdr-agile (always) and SoapyFobosSDR (optional) for
# aarch64 via Docker buildx (QEMU emulation). Produces dist/fobos-aarch64/ —
# a self-contained artifact embedded-agent can COPY into an OrangePi-targeted
# Dockerfile.
#
# Prerequisites:
#   docker buildx installed (Docker 19.03+)
#   QEMU binfmt registered:  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
#
# Usage:
#   ./setup/aarch64/build-docker.sh                       # WITH_SOAPY=0 (default)
#   ./setup/aarch64/build-docker.sh --with-soapy          # WITH_SOAPY=1
#   ./setup/aarch64/build-docker.sh --no-cache            # force clean rebuild
#   ./setup/aarch64/build-docker.sh --with-soapy --no-cache

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/dist/fobos-aarch64"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.build-aarch64"
IMAGE_TAG="fobos-aarch64-build:latest"

WITH_SOAPY=0
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --with-soapy) WITH_SOAPY=1 ; shift ;;
        --no-cache)   EXTRA_ARGS+=("--no-cache") ; shift ;;
        *)            EXTRA_ARGS+=("$1") ; shift ;;
    esac
done

echo "=== fobos aarch64 builder ==="
echo "output:      ${OUTPUT_DIR}"
echo "WITH_SOAPY:  ${WITH_SOAPY}"

# Verify buildx and QEMU
if ! docker buildx version &>/dev/null; then
    echo "ERROR: docker buildx not available. Install Docker 19.03+ or 'docker-buildx-plugin'."
    exit 1
fi

if ! docker buildx inspect default | grep -q "linux/arm64"; then
    echo "INFO: registering QEMU binfmt for arm64..."
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
fi

# Build the image (linux/arm64 emulated via QEMU)
echo "--- building (this takes ~5-10 min on first run due to QEMU emulation) ---"
docker buildx build \
    --platform linux/arm64 \
    --file "${DOCKERFILE}" \
    --target export \
    --build-arg "WITH_SOAPY=${WITH_SOAPY}" \
    --output "type=local,dest=${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    "${REPO_ROOT}"

echo ""
echo "=== build complete ==="
echo "artifact: ${OUTPUT_DIR}"
echo ""
echo "--- contents ---"
find "${OUTPUT_DIR}" -type f | sort
echo ""
echo "--- VERSIONS ---"
cat "${OUTPUT_DIR}/VERSIONS" 2>/dev/null || true
echo ""
echo "--- watchtower-edge Dockerfile snippet ---"
if [ "${WITH_SOAPY}" = "1" ]; then
cat << 'SNIPPET'
# WITH_SOAPY=1 build — full SoapyFobosSDR fallback path included.

# Install SoapySDR runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsoapysdr0.8 \
        soapysdr-tools \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install fobos libs + headers + SoapyFobosSDR module
COPY dist/fobos-aarch64/lib/     /usr/local/lib/
COPY dist/fobos-aarch64/include/ /usr/local/include/
RUN ldconfig

# (optional) udev rules — only needed on bare-metal, not inside container
# COPY dist/fobos-aarch64/udev/fobos-sdr.rules /etc/udev/rules.d/

# Verify SoapyFobosSDR module registration
RUN SoapySDRUtil --find="driver=fobos" 2>&1 || true
SNIPPET
else
cat << 'SNIPPET'
# WITH_SOAPY=0 build — pfobos native backend only, no SoapySDR runtime needed.
# This is the default after the pfobos-as-primary-sdr-backend cutover.

# Install fobos runtime libs (libusb-1.0-0 is the only system dep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install fobos libs + headers
COPY dist/fobos-aarch64/lib/     /usr/local/lib/
COPY dist/fobos-aarch64/include/ /usr/local/include/
RUN ldconfig

# (optional) udev rules — only needed on bare-metal, not inside container
# COPY dist/fobos-aarch64/udev/fobos-sdr.rules /etc/udev/rules.d/

# Verify libfobos loads via the pfobos wrapper
RUN python -c "from pfobos import FobosSDR; FobosSDR()" || true
SNIPPET
fi
