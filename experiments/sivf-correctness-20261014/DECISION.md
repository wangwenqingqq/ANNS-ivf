# Decision (2026-10-15)

**Stop throughput work; retain independently reproducible correctness failures.**
This is an audit checkpoint, not completion of every item in the multi-day plan.
No core algorithm, GTS++ project, other user's process, or host configuration was
changed.

The separate independent re-run in `RECHECK.md` confirms the same decision after
evidence-checker hardening. Its scheduling-dependent counts are retained
separately rather than substituted into the original tables.

## Required decisions

1. **Can the original implementation pass T1–T3?** No under these frozen
   contracts. T1 overwrites an undeleted ID; T2 recycles a reachable slab; T3
   creates orphan slabs in all 3000 ample-pool uninstrumented trials.
2. **What is source-only versus reproduced?** R1–R3 have synchronized GPU state
   counterexamples and search/resource checks. Public-API R4 is reproduced under
   explicit pool-unavailability injection. T5 has primitive-level injection
   evidence only. The fixed `shared_query[256]` risk remains source-confirmed but
   **not reproduced** by the 128/256/257/960 exact-search/memcheck cases.
3. **Is CAS-related leakage confirmed?** Yes, as preallocated pool-capacity loss.
   Count-only causality agrees with resource snapshots; the 1000-batch growth
   run leaves 94000 orphan slabs. Batch-dependent ranges are measured only for
   this GPU, compiler, source, input, and launch contract.
4. **How much throughput remains after correctness guarantees?** Unknown. No
   corrected implementation or valid Faiss comparison was measured. Do not use
   the upstream hard-coded baseline or this audit's trace durations to answer it.
5. **Is a new independent systems mechanism established?** No. Reproducible bug
   fixes are not by themselves a novel systems thesis. A future direction first
   needs a nearest-prior-art kill test and evidence of a general safety/layout/
   reclamation-cost conflict, followed by a non-incremental mechanism.

## Next checkpoint and reopen conditions

- Preserve the original source and all negative evidence before any repair.
- Finish full-index publication-order, concurrent duplicate-delete, same-ID
  update, and ABA coverage. Specify permitted query semantics and slab lifetime
  before treating a concurrent run as correct.
- Any Correct SIVF prototype must be separate from this baseline, define slot
  reservation versus live count, unlink/read-side lifetime, free-stack
  publication, and failed-CAS reclamation, and regress T1–T7.
- Reopen throughput work only for workloads that pass the declared correctness,
  sanitizer, and sustained resource-conservation gates; do not infer universal
  failure of the paper or universal safety from this bounded audit.
