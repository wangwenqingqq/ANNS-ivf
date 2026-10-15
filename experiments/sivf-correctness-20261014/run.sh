#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
out=${1:-"$here/runs/$(date -u +%Y%m%dT%H%M%SZ)"}
gpu=${GPU:-3}
test ! -e "$out" || { echo "Refusing to overwrite existing evidence: $out" >&2; exit 2; }
mkdir -p "$out/logs" "$out/snapshots"
out=$(cd "$out" && pwd)
cp "$here/SOURCE_COMMIT.txt" "$out/SOURCE_COMMIT.txt"
git -C "$root" diff --exit-code "$(cat "$here/SOURCE_COMMIT.txt")" -- \
  . ':(exclude)experiments'
exec 9>"$here/.gpu$gpu.lock"
flock -n 9 || { echo "Campaign GPU lock is busy" >&2; exit 2; }
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
if grep -F "$uuid" <<< "$apps"; then
  echo "GPU $gpu has an active compute process; no run started" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$out/harness"
cp "$here/build.sh" "$here/run.sh" "$out/harness/"
cp -r "$here/tests" "$out/harness/"
here="$out/harness"
export BUILD_DIR="$out/build" BUILD_LOG_DIR="$out/logs"
{
  date -u +%FT%TZ
  git -C "$root" rev-parse HEAD
  nvidia-smi -i "$gpu" --query-gpu=uuid,name,driver_version,compute_cap,pstate,power.limit --format=csv
  "${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" --version
  c++ --version | head -1
  python3 --version
} > "$out/logs/environment.txt"
(cd "$root"; git ls-files -z -- faiss CMakeLists.txt cmake | xargs -0 sha256sum) > "$out/logs/upstream_sources.sha256"
bash "$here/build.sh"
COUNTED=1 bash "$here/build.sh"
PAUSED=1 bash "$here/build.sh"
python3 "$here/tests/check_state.py" --self-test
python3 "$here/tests/test_evidence.py"
for case in smoke T1_hole T1_tail T2_head T2_middle T3 T3_growth T4 T5 T7_pool; do
  "$BUILD_DIR/test_correctness" "$case" "$out/snapshots/$case" 1000 2>&1 | tee "$out/logs/repro_$case.log"
done
"$BUILD_DIR/test_correctness_counted" T3 "$out/snapshots/T3_counted" 1000 2>&1 | tee "$out/logs/repro_T3_counted.log"
"$BUILD_DIR/test_correctness_paused" T5 "$out/snapshots/T5_injected" 10 2>&1 | tee "$out/logs/repro_T5_injected.log"
if [[ -f "$root/build-sivf-audit/faiss/libfaiss.a" ]]; then
  "${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" -std=c++20 -arch=sm_120 -I"$root" -Xcompiler=-fopenmp \
    "$here/tests/test_public_api.cu" "$root/build-sivf-audit/faiss/libfaiss.a" \
    -lcublas -lcudart -lopenblas -o "$BUILD_DIR/test_public_api"
  sha256sum "$root/build-sivf-audit/faiss/libfaiss.a" > "$out/logs/full_library.sha256"
  "$BUILD_DIR/test_public_api" 2>&1 | tee "$out/logs/repro_T7_public.log"
else
  echo "Unresolved: full Faiss build required for public API T7" | tee "$out/logs/repro_T7_public.log"
fi
for dim in 128 256 257 960; do
  set +e
  compute-sanitizer --tool memcheck --error-exitcode 99 "$BUILD_DIR/test_correctness" \
    "dim_$dim" "$out/snapshots/dim_$dim" > "$out/logs/memcheck_dim_$dim.log" 2>&1
  rc=$?
  set -e
  echo "dim=$dim exit=$rc" | tee -a "$out/logs/sanitizer_status.txt"
done
(cd "$out"; find build -maxdepth 1 -type f -name 'test_*' -print0 | sort -z | xargs -0 sha256sum) > "$out/logs/binaries.sha256"
(cd "$out"; find harness -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) > "$out/logs/harness.sha256"
python3 "$here/tests/summarize.py" "$out"
echo "Evidence retained in: $out"
