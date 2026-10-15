# Query work after churn: CPU attribution, not a GPU frontier

**Ordinary hole reuse plus an early occupancy test preserves the exact same live
candidate distances as Fused Dense. The residual here is extra slot metadata
inspection and footprint, not unavoidable dead-vector distance evaluations.**
Cross-hot migration also changes the live candidate work needed by fixed routing.
None of these observations admits a new mechanism or establishes GPU speed.

## Setup and evidence

- Parent `a139d04c17ea8dbf9cbef65f3f3cb5ffa812cad3`; public mirrored SIFT descriptors,
  FP32 128D, 64 trained routing lists, k=10. The [data amendment](AMENDMENT.md)
  records unavailable official-archive access and the disjoint training split.
  Byte identity with the complete official archive is not independently verified.
- 32,768 / 65,536 live source IDs; 64 epochs x 512 successful logical replacements
  per case, one fixed seed. This is respectively 1 / 0.5 live-set turnover, not
  a long-running service. Source IDs are unique; descriptor values may have ties.
- Random, FIFO and cross-hot update traces; original and selected-hot query
  traffic. Actual descriptor inputs, **synthetic update/traffic patterns**.
- At epochs 0/16/32/64, 64 query instances per traffic group; original nprobe
  grid 1/2/4/8/16/32/64, plus an explicitly post-observation exploratory integer
  calibration. [CONTRACT.md](CONTRACT.md) and [PROBE_CALIBRATION.md](PROBE_CALIBRATION.md)
  distinguish the two rather than retrospectively claiming preregistration.
- All representations use the same successful updates, live IDs, fixed routing,
  candidate sets and tie order. Fused is the competent primary control, not a
  pack-then-insert baseline or broken original SIVF.
  Payload arrays are immutable inputs: the replay mutates ID/slot states only.
  Payload movement/write bytes are derived layout-plan counts, not bytes actually
  copied or hardware traffic measured on either CPU or GPU.

## What remains after strong controls

The table shows total moved survivors and **final** extent rows. Both controls
write the same 32,768 inserted rows. Hole reuse moves zero surviving rows in the
modeled non-relocating per-list storage; allocator growth/copy is excluded for
both, so this is not a measured write-bandwidth or allocated-memory comparison.
Metadata ratios compare inspected hole slots to dense live candidates at the
supplemental minimum integer nprobe reaching mean Recall@10 >=0.90. The same
nprobe and actual recall apply to both representations within each comparison.

| Live rows | Trace | Fused moved / inserted rows | Final hole extent / live | Original-query metadata amplification | Hot-query metadata amplification |
|---:|---|---:|---:|---:|---:|
| 32,768 | random | 6,507 / 32,768 | 35,133 (1.072x) | 1.084x | 1.061x |
| 32,768 | fifo | 9,828 / 32,768 | 35,863 (1.094x) | 1.110x | 1.099x |
| 32,768 | cross-hot | 12,829 / 32,768 | 45,541 (1.390x) | 1.303x | 1.108x |
| 65,536 | random | 6,539 / 32,768 | 68,091 (1.039x) | 1.035x | 1.045x |
| 65,536 | fifo | 10,019 / 32,768 | 69,425 (1.059x) | 1.072x | 1.060x |
| 65,536 | cross-hot | 15,235 / 32,768 | 80,845 (1.234x) | 1.173x | 1.051x |

- Fused survivor rewrites are 19.86%–46.49% of inserted row writes across these
  traces. FP32 payload **write bytes** are rows x 128 x 4; this is neither
  read+write memory traffic nor end-to-end update overhead. Zero survivor moves
  in the hole model is not automatically a total-operation or GPU speed win.
- Early validity testing has live-distance amplification exactly **1.0x** against
  dense, with every declared selected-list top-10 checked from physical hole
  contents. The late-validity control charges distances to all traversed slots
  and therefore has the same amplification as metadata. It is an artificial
  diagnostic, **not an observed implementation defect** in SIVF/cuVS/Jasper.
- For cross-hot original traffic, the supplemental minimum integer nprobe at
  mean recall >=0.90 changes 4->5. Mean live candidates increase 1.338x / 1.345x
  at 32K / 65K, not the 1.861x / 1.920x obtained at the coarse 4->8 grid points.
  Actual final recall is 0.94375 / 0.9234375; this is a matched target, not exact
  equal achieved recall or a lower bound. No latency ratio follows from it.
- The hot-query cross-hot result is materially different: corresponding live
  candidate ratios are 1.555x / 1.045x. Random original-query traffic is 0.976x /
  1.008x. Churn alone does not imply monotonic query degradation.

`RESULTS.json` retains both curves, all 384 epoch ledgers, list high-water rows
and the two recall targets (0.90, 0.95). `QUERY.jsonl` retains all 3,072 query
instances, ground truth, original-grid returned IDs, probe ranks and full live /
hole prefix counts. Work P99 is a quantile of **inspected rows**, not latency.
Extents/high-water rows are not allocator capacity, reader-protected residency,
fragmentation bytes or peak VRAM.

## Validation and preserved rejected stages

- 1,275 exhaustive small fused plans match the pinned parent construction and
  the minimum moved-survivor count for that final dense-prefix contract.
- 20x100 randomized representation events; per-epoch live-ID/owner/unique-slot
  checks for both layouts; tail trimming/hole reuse and physical-query regression.
- Invalid-ID and duplicate-sparse-ID negative mutations are rejected. Matrix L2
  agrees with direct integer subtraction on the declared synthetic and real
  samples; nonfinite matrices are rejected. This is bounded validation, not
  a proof of arbitrary concurrent behavior or every numerical implementation.
- Remote CPU: Python 3.12.3, NumPy 1.26.4. Independent local CPU: Python 3.9.6,
  NumPy 2.0.2. All logical results, centroid/assignment hashes, input hashes and
  raw query receipts match exactly, except recorded software-version fields.
  The local NumPy run emitted matmul divide/overflow/invalid FP-status warnings
  despite passing finite and exact-integer checks. Their cause is **undiagnosed**;
  do not claim cross-platform numerical issues have been eliminated. Raw local
  logs are retained outside Git; the intended remote run has no such warnings.
- `primary_01`: harness tuple-indexed a dictionary, failed before any query or
  update result; source and failed evidence preserved, not admitted.
- `primary_02`: completed but lacked sparse duplicate-ID rejection; audit witness
  independently reproduced. No generated-state duplicate was established.
  Preserved as an incomplete validation checkpoint, not selected as final.
- `primary_03`: stronger invariants and mutation test, original probe grid only.
- `primary_04`: same original-grid results plus supplemental probe calibration;
  original fields/receipts were checked unchanged. Final source and evidence
  are bound by `MANIFEST.json`; historical run source/results remain ignored.

## Gate 0 and decision

[Quake](https://www.usenix.org/system/files/osdi25-mohoney.pdf) already adapts
partition maintenance and query scanning to workloads; [Ada-IVF](https://arxiv.org/abs/2411.00970)
already targets partition imbalance/drift with local maintenance. Generic
workload-aware compaction, early validity checks, sparse slots or reclustering
are **not** a new thesis. These papers are nearest-art controls, not reproduced
performance baselines in this experiment.

**Implementation:** CPU oracle/ledger validated within this finite contract;
GPU concurrency, API failure propagation, allocator admission and service gates
remain unknown. A logical zero-failure ledger does not test API OOM handling.
**Mechanism:** reject 'historical holes inherently require dead-vector distances'
under early filtering; retain metadata/extent versus survivor-copy trade-offs.
**Thesis:** no new mechanism or Pareto improvement admitted. Do not scale an
ordinary filter/compaction idea into a paper. The next admissible GPU question
is maintenance/publication cost under real readers, against competent existing
scoped-snapshot/EBR and matching ANN baselines, only after a GPU is allocated.
Reopen a representation proposal only for a concrete non-incremental mechanism,
nearest-art distinction, and matched-quality/semantics/budget evidence.

| Claim | State | Allowed scope |
|---|---|---|
| Plain early occupancy filtering can avoid dead-vector distances | partial | Same exact candidates in this serialized CPU representation; not a latency result |
| Cross-hot migration changes fixed-routing query work | partial | One seed, two finite SIFT-prefix scales; actual recalls and counterexamples retained |
| Hole extents equal allocated or reader-protected VRAM | rejected | Allocation, growth copies and concurrent retention are outside the model |
| A GPU sustainable-churn frontier is improved | unknown | No GPU execution, latency/throughput measurement or new protocol |

## Reproduce

Python 3.9+ and NumPy 1.22+; no CUDA, PyTorch, Faiss or GPU initialization.
`INPUTS.json` lists pinned public URLs and prefix hashes. Download only the
required byte ranges (verify returned sizes/hashes; some servers ignore Range):

```sh
EXP=experiments/query-work-20261015
BASE_URL=$(python3 -c 'import json;print(json.load(open("experiments/query-work-20261015/INPUTS.json"))["base_url"])')
QUERY_URL=$(python3 -c 'import json;print(json.load(open("experiments/query-work-20261015/INPUTS.json"))["query_url"])')
curl -fL --range 0-139493375 "$BASE_URL" -o base_prefix.fvecs
curl -fL --range 0-515999 "$QUERY_URL" -o query_prefix.fvecs
python3 "$EXP/check.py" --self-test
python3 "$EXP/check.py" --base base_prefix.fvecs --queries query_prefix.fvecs --out /tmp/query-work-fresh
cmp /tmp/query-work-fresh/QUERY.jsonl "$EXP/QUERY.jsonl"
```

Use a fresh output directory; the runner refuses overwrite. With the same
recorded runtime, `RESULTS.json` should match byte-for-byte. Across runtimes,
compare all results apart from `python`/`numpy` fields and retain any warnings
or assignment/hash differences rather than silently tolerating them.
Data, failed-run logs and machine configuration stay outside the publish set.
No GPU was launched or polled; previous reader-lifetime GPU gates are unchanged.
