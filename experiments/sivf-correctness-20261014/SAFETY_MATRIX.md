# Safety matrix (2026-10-15)

All original-algorithm results use pinned upstream commit
`c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`, one SM 12.0 GPU, and the contract
in `CONTRACT.md`. `EVIDENCE_RUN.txt` selects the raw run; filenames below are
relative to it. Passing a subset is not a safety proof.

| Test | Scope and repetitions | Result | Raw evidence | Remaining boundary |
|---|---|---|---|---|
| Phase 0 | Original full Faiss build; minimal three-source CUDA harness; 32 records, D=128, one query | Built; smoke state and exact top-k pass | Original full-build log; `snapshots/smoke/` | No full upstream test suite run |
| T1 partial hole | 32 vectors, delete observed slot 5, insert ID 100; one main-run sequence | **Reproduced** I1/I2/I5 and exact-search failure | `snapshots/T1_hole/`, `logs/repro_T1_hole.log` | Public-API T1 variant not tested |
| T1 matched control / I6 | Delete observed slot 31; repeat deletes in both cases | State/search control passes; repeated deletes return zero and leave equal snapshots | `snapshots/T1_tail/`, both `B_repeat.json` files | Concurrent duplicate deletes/ABA not tested |
| T2 head and middle | 64/96 vectors respectively, two lists, reclaim then reuse; one sequence each | **Reproduced** reachable/free overlap, cross-list sharing, and lost live records | `snapshots/T2_head/`, `snapshots/T2_middle/` | Reader-side reclamation protocol not proved |
| T3 original | Batch 64/256/1024; pool=64×batch; 1000 independent trials each | **Reproduced** orphans in 3000/3000; every attempted ID commits with this pool margin | `snapshots/T3/trials.csv` and first snapshots | Other GPUs/toolkits/shapes not tested |
| T3 counted | Same matrix, separate count-only build | Failed head CAS equals orphan count in every trial; zero reclamations | `snapshots/T3_counted/trials.csv` | Instrumentation changes scheduling; not the original-run rate |
| T3 accumulating | 1000 batches ×64 vectors; fixed 131072-slab pool | **Reproduced** growth to 94000 orphans; 64000 live records, 2000 reachable slabs | `snapshots/T3_growth/` | Long-running service VRAM/latency not measured |
| T4 natural overlap | 1000 trials; 32 old records, 8 inserts, 8 deletes, 32 queries; three nonblocking streams, common start gate | No inconsistent **returned** ID/distance pairs; captured quiescent states still fail pool conservation | `snapshots/T4/overlap.csv` and two snapshots | Does not inspect every intermediate payload; versions, duplicate-delete/update/ABA stress not implemented |
| T4 kernel-overlap diagnostic | Separate 10-trial Nsight Systems capture | CUPTI confirms search/insert overlap 10/10, search/delete 7/10, all three kernels 6/10 | `logs/T4_overlap.nsys-rep`, `.sqlite`, `OVERLAP_TRACE_SUMMARY.json` | Profiler schedule differs; no global-snapshot or performance claim |
| T5 natural primitive | Actual allocator `free_slab`/`allocate_slab`, two streams; 1000 trials | **Not reproduced**: zero stale slab returns | `snapshots/T5/publication.csv` | Primitive-level setup, not live-index Delete/Insert |
| T5 injected primitive | Copied original allocator header plus bounded publication pause; 10 trials | **Reproduced under injection**: stale held slab returned in 9/10 | `snapshots/T5_injected/publication.csv` | Full-index natural/injected publication interleavings remain unvalidated |
| T6 dimensions | D=128/256/257/960; 32 records, one exact top-10 query and one memcheck process per dimension | **Not reproduced**: all searches pass, each memcheck reports 0 errors | `logs/memcheck_dim_*.log`, `snapshots/dim_*/` | Source array-bound risk remains; no general high-D/concurrency guarantee |
| T7 raw pool | One slab, 32 successful inserts then ID 100 | **Reproduced**: 33 attempted, 32 committed | `snapshots/T7_pool/`, `logs/repro_T7_pool.log` | Not a public `ntotal` observation |
| T7 public API | 32 successful public inserts, then deliberately unavailable free stack and one more public insertion | **Reproduced under exhaustion injection**: invalid new ATT, no exception, `ntotal` 32→33 | `logs/repro_T7_public.log` | Not naturally exhausted large public-index workload |

## Explicitly uncompleted gates

The full multi-day plan is **not complete**: versioned GPU payload instrumentation,
adversarial same-ID update/ABA interleavings, full-index free-stack publication
tests, broad sanitizer/stress coverage, a corrected implementation, and valid
performance comparisons remain pending. No source-risk-only item is silently
upgraded to a runtime failure, and no narrow pass is upgraded to a proof.
