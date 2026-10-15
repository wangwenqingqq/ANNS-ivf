#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
source_dir="$here/external/MVGpuBTree"
pin=f755006d78ccca565f97cda053727d4d48b82d46
if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p "$here/external"
  git clone https://github.com/owensgroup/MVGpuBTree.git "$source_dir"
  git -C "$source_dir" checkout "$pin"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
mkdir -p "$here/build"
"${CUDACXX:-/usr/local/cuda/bin/nvcc}" -O3 -std=c++20 -arch=sm_120 -lineinfo \
  --expt-relaxed-constexpr -Xptxas=-v -I"$source_dir/include" "$here/probe.cu" \
  -o "$here/build/probe" 2>&1 | tee "$here/build/build.log"
sha256sum "$here/probe.cu" "$here/build/probe" "$source_dir/include/memory_reclaimer.hpp" > "$here/build/hashes.sha256"
