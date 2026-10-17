# Decision: stop exact 32-bit query-mask sharing

**No CUDA implementation or paper promotion.** The tested sharing mechanism
preserves integer scores but removes almost none of the first-filter arithmetic
on the held-out inputs. This is a negative mechanism experiment, not a claim
that the research brief's GPU acceptance criteria were completed.

## Exact work result

Batch32, scaling multiplier1.0, 64 held-out queries per dataset. Values are
candidate/list-size-weighted operation counts, not time or physical traffic.

| Dataset / selected base | Routing nprobe / global routing R@10 | Shared/native POPC | Shared/strong local CSE | Reconstruction updates/native | Added descriptor bytes, all64 queries |
|---|---:|---:|---:|---:|---:|
| SIFT32k | 16 / .99844 | .9989 | .9989 | 1.0000 | 115.6kB |
| SIFT262k | 16 / 1.00000 | .9989 | .9989 | 1.0000 | 115.6kB |
| GIST16k, 960D | 16 / .99063 | .9972 | .9973 | .9999 | 860.3kB |
| Gaussian8k, synthetic | 64 / 1.00000 | .9997 | 1.0000 | .9997 | 3355.4kB |
| GloVe32k, word-semantic | 64 / 1.00000 | 1.0000 | 1.0000 | 1.0000 | 447.5kB |

Thus the mechanism removes at most about0.28% native POPC at the primary scale,
not tens of percent. Across the predeclared .9/1/1.1 scaling sensitivity grid,
the largest reduction remains below0.4%. Batch1/8 show no material escape hatch.
All rows and both raw identity/complement ablations remain in the receipts.

**Cause:** two queries can share code loads without having identical 32-bit
query masks. Most word/plane masks remain distinct, including after complement
canonicalization. More database rows do not create mask repetition: the larger
SIFT case weights the same query information with more candidates. The exact
reconstruction work remains essentially unchanged; dictionaries and scheduling
are extra costs. It would be misleading to attribute the separately modeled
code-load reuse opportunity to this failed arithmetic-sharing hypothesis.

The ordinary load-sharing ratio is only a comparison against independent
logical source loads. The native implementation already has L2-oriented pair
ordering, and cache behavior is unmeasured. Any later tiling candidate needs a
known load-sharing baseline without CSE, not just the original query-major code.

## Gates reached and missing

- `measured`: CPU count matrix on the new Xeon Platinum8575C, five datasets,
  batches1/8/32 and three fixed scale multipliers; no retained CPU timing claim.
- `measured`: 5,701,602 integer-output comparisons on real/synthetic data words;
  independent dense integer products and native four-plane arithmetic agree.
  Six regression tests include negative mutations and a duplicate-query positive
  control. The latter proves the probe detects repetition when it exists; it is
  not a representative benchmark win.
- `partial`: source-shaped query codes use CPU PCG64 calibration and the old
  Faiss rotation. Neither GPU-generated codes nor kernel execution are reproduced.
- `unknown`: actual bottleneck shares, physical bytes, register/occupancy effects,
  query/grouping time, global pipeline recall, peak VRAM, P50/P99/QPS and sustained
  GPU behavior. An exact integer result does not fix threshold-observation order
  under a different grouped schedule.

GT is exhaustive over the entire **selected** base prefix, not full-million GT.
Routing recall in the table is not GPU index recall. GloVe adds word-semantic
coverage, not modern sentence embeddings. No policy was selected on evaluation
queries. The enlarged SIFT input was independently checked against its pinned
public mirror; prior data hashes and the GloVe archive/array hashes are retained.

## Research decision and reversal condition

Implementation status: CPU exact-integer prototype correct in the tested scope.
Mechanism status: **rejected for this frozen exact32-bit mask-sharing matrix**.
Thesis status: no new non-incremental mechanism or performance advantage shown.
Generic CSE/complement identities and quantized-computation reuse are established
ideas (MECHANISM.md); they must not be repackaged as a new ANN contribution.

Reopen only for an independently justified workload with substantial mask
repetition, such as genuine near-duplicate-query bursts, and only if it beats
local CSE after metadata/reconstruction/grouping costs. Changing to tiny
subwords and ignoring the multiplied lookup count is not a valid rescue.
This result does not reject approximate query representations, native tensor
operators or every multi-query organization; those are different hypotheses.

## The one next question

**Which native path actually dominates end-to-end GPU time: the always-paid
first filter at k10, or the full-ex-code fallback at k100?** The source exposes
both, but the timing shares remain unknown. There is currently no mechanism
qualified for promotion; this measured bottleneck question should decide the
next direction rather than preselecting another pruning structure.

If k10 is filter-dominated, investigate a genuinely different query/low-bit
execution contract against known bit-serial/tiled controls. If k100's fallback
is dominant, establish a fair tuned high-k baseline before claiming novelty;
changing a template limit is not a thesis. Preserve global quality, total query
preparation and memory budgets in either case. No GPU has been allocated or
accessed by this round, and no GPU launch is implied by this document.

The first primary run is retained unchanged. A second run replaces only the
hardcoded CPU-label metadata with live CPU discovery; it does not change the
mechanism, data, score arithmetic or decision rule. The comparison passes: all42,339 group rows are byte-identical, and all result
fields except the declared source hash and CPU-label metadata are identical.
Both raw run hashes are retained in MANIFEST.json.
