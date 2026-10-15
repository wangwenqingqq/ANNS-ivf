# Independent correctness recheck (2026-10-15)

**Decision: T1–T3 fail again; retain the audit checkpoint and keep throughput closed.**
The original index algorithms are unchanged. This is not a corrected index,
a completed concurrency proof, or a performance result.

## Contract and provenance

- Source: upstream `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`, verified against the live upstream main revision.
- Hardware/software: the same single SM 12.0 GPU, CUDA 13.1.115, driver 590.48.01; no compute process on the selected GPU at preflight and no host/power/process changes.
- Inputs, oracle, assignments, shapes, and stop rule: unchanged from `CONTRACT.md`.
- Run: `runs/independent_20261015`, separate from `runs/validation_20261015`.
- Evidence: `INDEPENDENT_SUMMARY.json` and `INDEPENDENT_MANIFEST.json`; 97 retained raw/harness/binary files independently re-hashed successfully.
- Build: all tracked upstream code outside the experiment tree matches the pinned commit. The original full Faiss target is up to date. The re-run builds all three minimal variants and the public-API harness, preserves their bytes, hashes the linked full library and upstream sources, and freezes the harness per run.
- Timing scope: none. No comparator or speedup claim.

## Observations

| Test | Independent result | Boundary |
|---|---|---|
| Smoke and T1 tail-delete control | State and exact top-10 pass | Passing controls, not universal index validation |
| T1 partial hole | Undeleted ID lost; two live IDs share ATT; count/bitmap and exact-search fail | Same original-algorithm transition as R1 |
| T2 head and middle | Reachable/free overlap, then cross-list sharing and 32 missing expected IDs | Same original-algorithm transitions as R2 |
| T3 original, batch 64 | Orphans in 1000/1000 trials; 94 per trial | Pool = 64 × batch; all 64 IDs commit |
| T3 original, batch 256 | Orphans in 1000/1000 trials; 922–1124 per trial | All 256 IDs commit; scheduling-dependent range |
| T3 original, batch 1024 | Orphans in 1000/1000 trials; 16338–16765 per trial | All 1024 IDs commit; scheduling-dependent range |
| T3 count-only | Every trial: orphan count equals failed head CAS; no reclamations or failed inserts | Separate instrumented scheduling, not substituted for original counts |
| T3 accumulating | 64000 live IDs, 2000 reachable slabs, 35072 free, 94000 orphan after 1000 × 64 inserts | Lost preallocated capacity, not additional `cudaMalloc` bytes |
| T4 bounded overlap | 0 inconsistent returned ID/distance pairs in 1000 trials | Event windows do not prove instruction overlap; no new CUPTI capture; earlier diagnostic remains separate |
| T5 allocator primitive | Natural: 0/1000 stale returns; injected: 9/10 | Not full-index publication interleavings |
| T6 D=128/256/257/960 | Exact search passes; four memcheck processes report 0 errors | Source bounds concern not reproduced; not a high-D safety proof |
| T7 public API | `ntotal` 32→33, invalid new ATT, no exception | Explicit unavailable-pool injection |

## Evidence-tool corrections

The read-only review found checker weaknesses, not contrary runtime evidence:

1. Require snapshot schema, dimensions, slab indices, a 32-bit bitmap, and exactly one serialized record per set bit. Reject duplicate oracle IDs and malformed coordinates/payload shapes.
2. Reject truncated top-10 labels/distances, nonfinite distances, missing required cases, and incomplete/reordered trial matrices. An empty evidence directory cannot produce a successful summary.
3. Keep controls, expected counterexamples, unresolved public-API scope, and index validation separate in the machine-readable assessment.
4. Hash only existing optional public-API binaries, so the documented minimal-only path can summarize an unresolved public API instead of aborting during hashing.
5. Preserve run-specific harness/binaries and broaden upstream source verification beyond four files. An unrelated GPU-query failure is fatal rather than mistaken for an idle GPU.

CPU checker/adversarial regressions and shell syntax checks pass. The earlier
74-file raw manifest remains valid and untouched; old numeric ranges are not
rewritten to match the new scheduling outcomes.

## Still pending

Versioned intermediate-payload checks, concurrent duplicate-delete/update/ABA,
full-index publication-order tests, broad sanitizers/stress, a separate Correct
SIVF prototype, and valid independent performance comparisons remain open.
Before a new systems mechanism or paper, run its nearest-prior-art novelty gate;
these implementation bugs alone do not establish a new research thesis.
