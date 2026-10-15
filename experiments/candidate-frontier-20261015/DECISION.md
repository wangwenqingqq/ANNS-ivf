# Decision: stop the known-certificate implementation path

Experiment `ivf_20261015_candidate_information_cpu_v1`, 2026-10-15.
Parent `c98871ad4217b3b11a68461d088d89d784481891`.

**Do not promote these controls into a new method or CUDA system.** There is
large ideal opportunity, but no demonstrated non-incremental mechanism or
advantage beyond mature quantization. This closes a bounded CPU opportunity
screen, not the full GPU research acceptance criteria.

## Symmetric hypothesis decisions

| Hypothesis | Evidence state / decision | Evidence, material limitation |
|---|---|---|
| H1: many candidates are irrelevant to exact refinement | `partial`: supported as an oracle opportunity | The free-selector read ratios are .01658 SIFT, .00386 GIST, .00330 synthetic Gaussian. Selected-list exactness is conditional on routing; global R@10 is .99844/.99063/1.0. The free selector is not implementable evidence. |
| H2: cheap information can discriminate candidates | `partial`: representation-dependent; these known controls are not a new direction | Online SQ8+error reads .28676/.27108/.27573 of Flat under the candidate-byte model. D/8 and block norms are mostly ineffective; D/2 only saves under the primary refetch model on SIFT. SQ8 uses D-dimensional arithmetic and additional storage. Native 4-bit RaBitQ+32 refinement matches all selected-list IDs for the two real subsets at lower modeled read ratios .18074/.13771 and lower static storage. No universal quantizer guarantee follows. |
| H3: execution granularity can erase selectivity | `partial`: work-model evidence only; physical GPU claim `unknown` | Expanding SQ8 survivors to 32-candidate groups raises read ratios to .33975/.38455/.55500. D/2 prefix on SIFT changes from .79660 to 1.25190. A group-expansion model is not a measured transaction, warp behavior or latency. |
| H4: query difficulty enables useful batch adaptation | `inconclusive`: some heterogeneity, no routing/scheduler claim | Gap-quartile SQ8 read-ratio means span .28624–.28763 SIFT and .26975–.27208 GIST; there is no consistent monotone difficulty advantage. Batch-32 max/mean refined work is about 1.23–1.31 SIFT and 1.39–1.46 GIST. These bounded proxies do not establish or reject an adaptive GPU representation. |

All 64 evaluation queries per real dataset, all four probes, and all five
certificate controls are retained. Ground truth is exhaustive over selected
prefixes. Only routing nprobe was calibrated; native refinement pools were not
quality-tuned on evaluation results. Full raw grids remain in the evidence.

## Negative-evidence ledger and reopen boundary

| Target / keeper | Rejected implementation path | Added cost / effect | Diagnosis | Reopen condition |
|---|---|---|---|---|
| Fixed candidate sets / FP32 Flat | Small raw prefixes as an immediately profitable generic filter | Evidence read plus payload refetch; D/8 costs 9.6–12.4% more reads at selected probes | Too weak even with final-threshold hindsight; not merely a threshold bootstrap issue | A new query-conditioned information mechanism, with its own cost included, changes the bound rather than only rewriting the same scan in CUDA |
| Same contract / Flat | 8 or 16 block norms as the main certificate | 64/128 stored bytes per vector; <=1.9% savings on real high-D GIST and regressions on SIFT/Gaussian | Norm evidence is too coarse for these distributions | Demonstrate another independently selected workload where a distinct mechanism—not a renamed norm bound—removes dominant work |
| Real-prefix native quantization controls | SQ8+residual as a new thesis | About 27–29% of Flat reads; more storage and modeled reads than 4-bit RaBitQ+32 at empirically identical real-query answers | Standard quantization/error-bound filtering is already established; not a novelty claim | Non-incremental mechanism and reproducible same-quality, same-budget advantage beyond the nearest strong baseline |

These findings do **not** reject every certificate, tree representation,
query-adaptive policy, probabilistic contract or candidate-information research.
They reject immediate promotion of the tested known constructions in this scope.
No result about concurrent versions, scoped snapshots, EBR, or tree replacement
is inferred from this unrelated static candidate screen.

## One next falsification question

**Can query-batch-shared information safely eliminate complete candidate tiles
*before reading their extended low-bit codes*, beyond the filtering already
present in GPU IVF-RaBitQ, without extra D-dimensional work or excess storage?**

This is a hypothesis, not a novelty assertion. First inspect the pinned GPU
IVF-RaBitQ implementation for the same tile-level work elimination. If it already
subsumes the mechanism, stop/switch. If not, the next cheap test should count
whole low-bit-code tiles avoided on the same held-out real queries while charging
shared-evidence generation and storage; a per-vector oracle alone cannot pass.
Only a materially positive, new mechanism warrants an allocated-GPU campaign.
No new experiment or GPU launch is authorized by this decision document itself.

## Explicit open gates

- GPU-native IVF-RaBitQ, DARTH and TurboQuant were not executed here.
- No GPU QPS, p50/p99, physical bytes, peak VRAM, sanitizer, profiler, sustained
  workload or end-to-end Pareto claim. Certificate update/encoding/preparation
  timing and native query LUT/rotation overhead are not closed.
- PQ training rows are below the library's recommendation; do not claim a tuned
  high-D PQ frontier or use that weakness to rescue a thesis.
- A larger, independently selected real-data sample, semantic embeddings, more
  training data and a finer nprobe grid could change quantitative boundaries.
  They are not reasons to relabel standard quantization as a new mechanism.

No paper prose or new GPU implementation is promoted. Allowed wording is the
bounded CPU information-cost result above, with all caveats attached.
