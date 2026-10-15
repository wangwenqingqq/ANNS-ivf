# Frozen correctness contract (2026-10-15)

- Source: `hpdic/ElasticIVF`, commit `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`.
- Candidate: none. Do not repair upstream allocation, deletion, search, or reclamation before observing failures.
- Hardware: one RTX PRO 6000 Blackwell Server Edition, SM 12.0, CUDA 13.1.
- Inputs: deterministic nonperiodic hash-derived FP32 vectors; explicit coarse assignment; no datasets or k-means.
- Oracle: host map of requested live IDs, owning list, and exact vector payload; full-scan L2 top-10 on the selected list.
- T1/T2 initialization: one-vector launches, synchronized, to isolate the state transition from allocation contention. Slot IDs are read from GPU state, not assumed from insertion order.
- T1: 32 vectors, delete observed slot 5, repeat the deletion, insert ID 100. Matched control deletes observed slot 31.
- T2: 64/96 vectors, delete the head/middle slab respectively, then insert ID 100 into a second list.
- T3 initial pilot: 64/256/1024 concurrent insertions into one list, 1000 independently reset trials each; pool capacity is four times the batch. The pilot is preserved separately and must not be presented as an ample-pool experiment if it exhausts the pool.
- T6: dimensions 128/256/257/960, 32 serially inserted vectors, one query, k=10. Run each sanitizer case in a separate process.
- T7: pool exhaustion is first observed at the raw kernel boundary; a public `ntotal` claim additionally requires the real public index API.
- Quiescent snapshots follow `cudaDeviceSynchronize`; intermediate reservation counts are not checked as if quiescent.
- Resource orphan: slab is neither reachable from any head nor on the active free stack. A failed CAS alone is not a failure criterion.
- Stop rule: any structural counterexample closes the throughput gate. Continue only bounded correctness/safety diagnosis; do not start large data, timing comparisons, a corrected implementation, or paper drafting.
- Timing scope: none. No speedup, throughput, or production-quality claim is admissible.
- Build scope: direct inclusion of the three original `.cu` files plus their original headers; unused Faiss host wrappers can be discarded by function-section linking. Full Faiss build is a separate gate.
- Execution safety: use physical GPU 3 only after checking its process list; take a local advisory `flock`; do not alter clocks, power limits, other processes, services, or host configuration. Release all allocations on process exit.

## Evidence states

`Source-confirmed`, `Reproduced`, `Not reproduced`, and `Unresolved` remain distinct. A low-level test is not a public API test; an injected schedule is not a natural-run observation; a sanitizer-clean subset is not a proof of concurrency safety.

## Predeclared amendment after the bounded-pool pilot

The 256/1024 pilots exhausted or approached exhaustion. Repeat the same 1000-reset-trial matrix with pool capacity **64 times the batch**, preserving the first matrix and its binary. Also run 1000 accumulating batches of 64 vectors in a pool of 131072 slabs with a 1024-slab safety margin; retain the per-batch resource ledger. Run count-only instrumentation separately (failed head CAS, allocations, reclamations, successful and failed inserts), without delay injection. No throughput denominator is introduced by this amendment.
