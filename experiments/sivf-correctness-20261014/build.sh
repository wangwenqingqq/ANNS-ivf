#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
build=${BUILD_DIR:-"$here/build"}
logs=${BUILD_LOG_DIR:-"$here/logs"}
mkdir -p "$build" "$logs"
suffix=""
flags=()
if [[ ${COUNTED:-0} == 1 ]]; then
  python3 "$here/tests/make_counted.py" "$root" "$build/counted_append.cu"
  suffix="_counted"
  flags=(-DSIVF_COUNTED -I"$build")
fi
if [[ ${PAUSED:-0} == 1 ]]; then
  test ${COUNTED:-0} == 0
  python3 "$here/tests/make_paused.py" "$root" "$build/paused/faiss/gpu/impl/SlabManager.cuh"
  suffix="_paused"
  flags=(-DSIVF_FREE_PAUSED -I"$build/paused")
fi
"${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" -std=c++20 -O2 -lineinfo -arch=sm_120 \
  "${flags[@]}" -I"$root" \
  -Xcompiler=-ffunction-sections,-fdata-sections -Xlinker=--gc-sections \
  "$here/tests/test_correctness.cu" \
  "$root/faiss/impl/FaissException.cpp" \
  -o "$build/test_correctness$suffix" 2>&1 | tee "$logs/build_minimal$suffix.log"
sha256sum "$build/test_correctness$suffix" | tee "$logs/binary$suffix.sha256"
