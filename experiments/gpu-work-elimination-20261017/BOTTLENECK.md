# Bottleneck frontier: paid work is not a measured latency share

**The first filter still computes each query/candidate word independently.**
The pinned source permits an exact integer-work experiment on this remaining
stage. It does not permit assigning GPU latency shares from CPU counts.
No GPU was accessed in this round. R1's actual timing decomposition and R3's
matched-quality QPS/P99/VRAM gates remain **unknown**, not completed.

## Comparator and active paths

GPU reference: [cuvs_rabitq at 33d1de70](https://github.com/Stardust-SJF/cuvs_rabitq/tree/33d1de70fba2bd86390ea5915c62917c3a85c2ff).
The following are source-level facts; no kernel was built or profiled here.

| Work / source location | What is already eliminated | What remains / measurement needed |
|---|---|---|
| Rotation, routing and pair construction: `ivf_gpu.cu:1512–1678` | Batch query preparation and a small-pair sort bypass | Query preparation, centroid selection and grouping are real costs. Small batches do not automatically use cluster-major ordering. |
| Defaults: `ivf_rabitq.hpp:178–230` | Quant4; centroid/representative seed, warmup, dynamic block/IP choices | Compare against these enabled controls, not an infinite-threshold strawman. |
| Main Quant4 filter: `searcher_gpu_quantize_query.cu:173–280` | Existing lower-bound filter rejects candidates before later exact-IP/ex-code work | Its inner loop still executes signed four-plane AND-popcounts over every 32D word for each candidate/query. No cross-query mask factorization is present in this path. |
| Candidate stages: same file, following the filter | Candidates are compacted before later work; IP mapping adapts to candidate count | POPC savings do not imply ex-code savings: the proposed integer result is unchanged. At fixed thresholds the filter decision is unchanged; a real grouped schedule can observe different atomic thresholds, so pipeline parity must be checked separately. |
| Global threshold and seeding: `searcher_gpu_common.cuh:211–287`, atomic updates in search kernels | Shared query threshold, representatives and warmup already exist | Proposing shared thresholds or first-list warmup is subsumed. |
| Selection dispatch: `searcher_gpu.cuh:24–26`, Quant4 host dispatch | Fused selection when k<=64 | k>64 takes another path: `computeInnerProductsWithBitwise<true>` precomputes extended-code IP for every vector before final selection. This source-level coverage cliff is not a measured GPU timing cliff or a new contribution. |

Links: [main filter and query preparation](https://github.com/Stardust-SJF/cuvs_rabitq/blob/33d1de70fba2bd86390ea5915c62917c3a85c2ff/cpp/src/neighbors/ivf_rabitq/gpu_index/searcher_gpu_quantize_query.cu),
[query/global seed](https://github.com/Stardust-SJF/cuvs_rabitq/blob/33d1de70fba2bd86390ea5915c62917c3a85c2ff/cpp/src/neighbors/ivf_rabitq/gpu_index/searcher_gpu_common.cuh),
[public defaults](https://github.com/Stardust-SJF/cuvs_rabitq/blob/33d1de70fba2bd86390ea5915c62917c3a85c2ff/cpp/include/cuvs/neighbors/ivf_rabitq.hpp).

The paper identifies filtering as important, but that is a reported observation,
not a fresh profile on this hardware/workload. Source inspection cannot establish
whether POPC issue, short-code/factor traffic, queue work, synchronization,
allocation or query preparation is now the dominant critical path.
[GPU IVF-RaBitQ paper](https://arxiv.org/abs/2602.23999).

## Quality caveat in the reference seed

Representative K-th distances alone bound the raw K-th distance, provided there
are at least K distinct representatives. The code additionally takes the minimum
with a centroid-distance-times-scale heuristic. That second term is not a
universal safe upper bound, regardless of a source comment suggesting otherwise.
Counterexample: database {-1,+1}, centroid0, query0, k1. The true squared distance
and representative bound are1; scaled centroid distance is0. Their minimum is0.
Treat this as an approximate policy requiring actual global recall calibration,
not a deterministic guarantee. The mask factoring test does not change it.

## What the cheap experiment can and cannot answer

For each real query/list incidence, `probe.py` counts native signed4-bit
AND-popcounts, a stronger local per-query CSE control, and exact batch-shared
mask plans. Counts are weighted by list length, not averaged over empty work.
Reconstruction updates and query-descriptor/live-state models are retained.
The candidate removes repeated integer operations; it deliberately does not
claim to eliminate additional candidate ex-code reads. This addresses a
remaining earlier stage rather than counting the same 1-bit filtering twice.

The CPU sample uses a source-shaped query quantizer with a different seeded RNG
and a preserved Faiss rotation; native GPU encoding and float-stage parity are
not established. The integer identity is valid for arbitrary masks, but the
measured repetition distribution is a bounded CPU opportunity screen. Scheduling
can change when the shared atomic threshold is observed even with exact integer
arithmetic. Neither identical candidate sets nor native end-to-end output parity
is established by this test.

## Ideal-elimination equation, with unknowns kept unknown

Let s be the **measured** end-to-end share attributable to the affected work, f
its realized time reduction, and c added cost as a fraction of baseline query
time. Then the serial Amdahl model is `1/(1-s*f+c)`. Neither s, f nor c has been
measured on GPU. A POPC count reduction is not f. Kernel overlap, memory behavior,
register pressure, scatter/reconstruction, GPU grouping and critical-path shifts
can invalidate a work-proportional estimate. No numeric speedup ceiling is
claimed from these counts, and there is no QPS/P99 plot.

## First allocated-GPU experiment, not executed

Freeze the exact native index, query masks, query order, global GT and source
pin. First profile the native k10 path at batch1/8/32 and the k100 fallback.
The rejected mask-sharing candidate is not scheduled for CUDA implementation;
reopen it only under the reversal condition in DECISION.md. Keep rotation, route, query
quantization, dictionary construction/grouping, seed/warmup, filter, all later
stages and merge in the end-to-end timer. Record filter POPC instructions,
short-code/factor traffic, grouping/scatter, registers/shared memory and complete
query P50/P99/QPS. A source-level k100 fallback requires a fair tuned comparator,
not a paper win from increasing a template limit. No launch instruction in this
document authorizes using an occupied GPU.

## CPU reproduction and evidence locations

Use Python3.12, NumPy1.26.4 and faiss-cpu1.15.1; h5py3.14.0 is needed only for
public GloVe preparation. Reproduce the previous stage's inputs/models first,
following its README; they are not downloaded or rebuilt implicitly here.

```sh
mkdir -p inputs
curl -fL https://ann-benchmarks.com/glove-100-angular.hdf5 -o inputs/glove.hdf5
python prepare_glove.py --source inputs/glove.hdf5 --out inputs/GloVe100.npz
python probe.py --previous "$PREVIOUS_FRONTIER_DIR" \
  --large-sift "$SIFT_BASE_FVECS" --glove inputs/GloVe100.npz \
  --out runs/reproduce_01/out
python -m unittest -v test_probe.py
```

`SIFT_BASE_FVECS` needs at least262144 rows from the pinned SIFT base mirror in
MANIFEST.json; an entire1M archive is unnecessary. Three prior datasets, the
larger SIFT prefix and GloVe remain distinct rows, not a selected aggregate.
Outputs refuse overwrite. The committed RESULTS.json/GROUPS.jsonl.gz retain the
full count matrix, not QPS measurements. The self-tests include a duplicate-query
positive control that really reduces mask work, a case where complement CSE
costs more, independent dense-integer parity and a seed-bound counterexample.
The raw run snapshots/stdout remain outside Git, with hashes in MANIFEST.json.
