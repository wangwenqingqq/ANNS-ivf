# SIVF correctness counterexamples

This is an independent correctness audit of upstream ElasticIVF commit
`c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`, not a repaired index or a new-system performance result.

**Decision:** the original implementation fails the partial-hole, slab-reclamation,
and allocation-conservation tests. Keep the throughput gate closed. See
[REPRODUCIBLE_ISSUES.md](REPRODUCIBLE_ISSUES.md),
[SAFETY_MATRIX.md](SAFETY_MATRIX.md), and [DECISION.md](DECISION.md).

An independent re-run reproduced T1–T3 after hardening the evidence checker.
See [RECHECK.md](RECHECK.md) for the separate run, exact scope, and remaining gates.

## Reproduce

Requires one otherwise unused NVIDIA SM 12.0 GPU, CUDA 13.1, C++20, Python 3,
`flock`, and Compute Sanitizer. The minimal harness directly compiles the original
append/search/deletion implementations without modifying their algorithms.

From the repository root:

```bash
EXP=experiments/sivf-correctness-20261014
bash "$EXP/build.sh"
GPU=3 bash "$EXP/run.sh"
```

`run.sh` refuses to overwrite evidence or run on a GPU with an active compute
process. It creates a timestamped directory under `runs/`. Expected correctness
counterexamples are recorded in JSON, not treated as successful index validation.
`check_state.py` returns exit code 1 on a violated invariant; malformed snapshots
are errors, never passes. A harness/runtime error aborts the main run; sanitizer errors are
retained per case so later dimensions can run in fresh processes.
Each new run retains its own frozen harness, generated variants, binaries,
source/library hashes, and environment record. The summary rejects missing
required cases and malformed/truncated evidence; `assessment` distinguishes
passing controls from reproduced counterexamples and never validates the index.

For the full Faiss build and the public-API failure-propagation test:

```bash
cmake -S . -B build-sivf-audit -G Ninja \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_HOME:-/usr/local/cuda}" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF -DFAISS_ENABLE_EXTRAS=OFF -DFAISS_ENABLE_MKL=OFF
cmake --build build-sivf-audit --target faiss -j 8
GPU=3 bash "$EXP/run.sh"
```

Full Faiss additionally requires OpenBLAS/LAPACK, OpenMP, MPI, CMake >=3.24,
and Ninja. Without `build-sivf-audit/faiss/libfaiss.a`, the runner labels the
public API test unresolved; raw-kernel failure testing still runs.

Single counterexample (choose a new output directory):

```bash
CUDA_VISIBLE_DEVICES=3 "$EXP/build/test_correctness" T1_hole /tmp/sivf-T1-new
python3 "$EXP/tests/check_state.py" /tmp/sivf-T1-new/C.json
```

The single-case binary assumes the selected GPU is available; unlike `run.sh`,
it does not itself reserve/check the device.

## Evidence

`EVIDENCE_RUN.txt` identifies the retained validation run on the execution
checkout. `EVIDENCE_SUMMARY.json` is derived from raw snapshots and CSV files;
`EVIDENCE_MANIFEST.json` records relative filenames, sizes, and SHA-256 hashes.
Large snapshots, raw traces/logs, and binaries remain outside Git. This repository
ships the reproducer, curated results, and manifests, not all raw evidence bytes.
Re-running generates independent evidence; exact scheduling-dependent counts need
not match the recorded ranges.

`COUNTED=1 bash "$EXP/build.sh"` generates a separate count-only append variant.
`PAUSED=1 bash "$EXP/build.sh"` generates a separate, bounded free-stack
publication-pause variant. Both are test-only copies; neither replaces upstream
source or the uninstrumented executable. The pause experiment tests allocator
primitives, not the complete live-index Insert/Delete path.

For optional diagnostic-only kernel-overlap verification, run `nsys profile
--trace=cuda --sample=none --cpuctxsw=none` around the `T4` case, export its report
to SQLite with `nsys export --type sqlite`, then run `tests/inspect_overlap.py`
on that database. Serialize this under the same GPU lock. Trace durations are
not a throughput or speedup denominator.

Not covered: concurrent duplicate deletes, same-ID updates/ABA, full reclamation
proofs, corrected SIVF, and any valid independent performance comparison.
