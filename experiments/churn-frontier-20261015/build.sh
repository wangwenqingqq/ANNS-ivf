#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$here/build"
"${CUDACXX:-/usr/local/cuda/bin/nvcc}" -O3 -std=c++20 -arch=sm_120 -lineinfo \
  -Xptxas=-v "$here/probe.cu" -o "$here/build/probe" 2>&1 | tee "$here/build/build.log"
sha256sum "$here/probe.cu" "$here/build/probe" > "$here/build/hashes.sha256"
