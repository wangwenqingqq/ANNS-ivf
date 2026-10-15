# Claim–evidence ledger

| ID | Claim | State | Evidence and exact scope | Allowed wording / counterevidence |
|---|---|---|---|---|
| C0 | COW/batch/epochs/indirection are new here | rejected | `PRIOR_ART.md`; primary sources, access 2026-10-15 | Not admitted as standalone novelty; no system thesis rescued by implementation detail |
| C1 | Fused packing removes survivor moves | measured | `SUMMARY.json`, N8192, random/hot, planned moves exercised by oracle-checked kernels | Exact tested shape counts only, not an all-index lower bound |
| C2 | Indirect improves the primary epoch estimator | rejected | `SUMMARY.json`, refined 12-shape/4-process matched contract | No geometric-mean improvement here; one individual process can differ; larger regimes untested |
| C3 | The diagnostic is correct | partial | Declared endpoint oracle, small memcheck/synccheck, bounded stresses; `evidence/GATES.json` | Only the named gates pass; not a concurrency or lifetime proof |
| C4 | Quality is matched | partial | Same synthetic queries/partition sets and exact CPU oracle | Not trained IVF or real-data quality; D768 is saturated |
| C5 | Sustainable churn / 90% static QPS / 2× dynamic gain | unknown | No same-scope external dynamic/static benchmark | No claim; GPU component spans exclude costly host planning |
| C6 | New reader-lifetime mechanism is justified | unknown | No new measurement yet; GPU scoped-snapshot prior work must be the control | A question for the next cheap kill test, not an admitted contribution |

All measurements are from the frozen sources/binaries identified in the summary
and raw manifests. Review date: 2026-10-15. Do not substitute profiler durations
for the event-timing denominator or promote these claims to real workloads.
