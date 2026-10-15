# Decision (2026-10-15)

## 1. Phenomenon — measured, narrow

Ordinary fusion removes 73.0%–90.5% of planned survivor moves in random/hot
cases, while the refined indirect layout has primary paired F/I epoch ratios
0.872–0.985. No tested shape's geometric-mean primary estimator improves.
Initial and refined runs both remain available; results are not original SIVF
performance. Exact scope, order splits and exceptions are in `RESULTS.md`.

## 2. Explanation — evidence versus inference

Source and oracle checks establish the competing data movement paths; the
same-order P/I scan contrast isolates addressing more cleanly than F/I. CUPTI
localizes substantial work to the scan/local-top-k stage, not an abstract
reclamation cost. It does not isolate cache traffic from selection work.
Inference: in this small-list regime, avoiding survivor moves alone is not a
useful sufficient hypothesis for a better query+churn operating point. Do not
generalize this to larger lists, other kernels, real drift or all indirect layouts.

## 3. Idea — no novel mechanism admitted

Keep fused packing as a competent diagnostic control. Do not package COW,
GPU epochs, batch ownership, generic adaptive maintenance or indirection as a
new system: the closest work already covers these broad ideas. A normal launch
optimization and a small update-only signal do not rescue a subsumed thesis.

The next falsifiable **problem hypothesis**, not an accepted design, is that
under truly overlapping readers/writers, *query-visible version lifetime* rather
than copying bytes becomes a principal memory/latency constraint. First compare
against the GPU scoped-snapshot/EBR work listed in `PRIOR_ART.md`. If its ordinary
implementation resolves the problem, switch questions rather than rename it.
Only a measured remaining gap can motivate a non-incremental layout/execution
mechanism. No new prototype should be called novel before that test.

## 4. Evidence — C/Q/U/M/X/N

| Axis | Current state | Scope / missing gate |
|---|---|---|
| C | partial | Declared quiescent checks pass; concurrent history/intermediate coverage unknown |
| Q | partial | Equal synthetic recall; trained real-data cuVS curve not measured |
| U | partial / not promoted | GPU component spans only; no full dynamic API throughput |
| M | partial | Fixed explicit pools and bounded reuse; online reader-retirement/peak VRAM unvalidated |
| X | unknown | No overlapping-query/update or service-tail claim |
| N | rejected as a thesis | Existing mechanisms; no nontrivial system Pareto improvement established |

**Decision: do not scale or write a paper around this indirect prototype.** Retain
the negative evidence and switch the next cheap test to reader-lifetime pressure,
with nearest-art controls and explicit observable semantics. Reopen payload
mobility only when larger-list or different-kernel evidence identifies a material
unavoidable relocation cost and the same-contract fused control is remeasured.
The brief's 90% and 2× targets remain untested at their intended system scope,
not silently converted into hard early termination criteria.
