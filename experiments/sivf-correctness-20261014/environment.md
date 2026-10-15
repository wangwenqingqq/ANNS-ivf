# Verified environment (2026-10-15)

| Item | Value |
|---|---|
| Upstream | `https://github.com/hpdic/ElasticIVF` |
| Pinned commit | `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546` |
| User fork | `https://github.com/wangwenqingqq/ANNS-ivf` (public fork) |
| Branch | `experiments/sivf-correctness-20261014` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0 |
| Device scope | One physical GPU, index 3; remapped to CUDA device 0 |
| Driver | 590.48.01 |
| CUDA compiler | 13.1.115 |
| Host compiler | GCC 13.3.0, C++20 |
| CMake | 3.28.3 |
| Python | 3.12.3 |
| Inputs | Synthetic deterministic FP32; contiguous `[vector, dimension]` |
| Dimensions | 8 for state transitions; 128/256/257/960 for dimension tests |
| Search | L2, full selected-list scan, k=10, nprobe=1 |
| Training/data downloads | None; assignments supplied directly; public API control uses one manually populated centroid |
| Clock/power changes | None |
| Execution isolation | No other compute process on the selected GPU at preflight; advisory `flock` |
| Full Faiss gate | Original source configured and all 300 build steps completed, `libfaiss.a` linked |
| Minimal gate | Three original CUDA implementation files compiled directly; unused wrappers discarded by function-section linking |

The initial CMake invocation found the `nvcc` PATH symlink but failed to locate
the CUDA library root. Explicit toolkit/compiler paths resolved this without
editing upstream build files or changing the host installation. The initial
full-history network clone was replaced with a verified shallow transfer of the
same pinned commit; no user files or other processes were replaced.

Build flags and run commands are in `build.sh`, `run.sh`, and `README.md`.
Exact binary/harness hashes are retained in the validation run's
`logs/binaries.sha256` and `logs/harness.sha256`. The independent re-run additionally
retains `logs/upstream_sources.sha256`, `logs/full_library.sha256`, and frozen
run-specific harness/binary bytes. The original full-build log remains in the
execution checkout's raw logs.

This is a correctness-only campaign. No clock-normalized, isolated-latency,
sustained-throughput, comparator, or end-to-end performance result is asserted.
