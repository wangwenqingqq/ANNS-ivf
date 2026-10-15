# CPU query-work attribution contract

Experiment: `ivf_20261015_query_work_sift128_cpu`.
Parent: `a139d04c17ea8dbf9cbef65f3f3cb5ffa812cad3`. Frozen before execution.
This is an explanatory operation-count experiment, not an admitted new mechanism.
No GPU execution, GPU polling, concurrent readers, CPU speed comparison, QPS,
latency P99, allocated-VRAM claim or sustainable churn measurement is allowed.

## Hypothesis and nearest art

After competent hole reuse, the residual trade-off may be survivor rewrites
versus metadata scans, rather than unavoidable invalid-vector distances.
A plain occupancy prefilter is a strong competing explanation for any claimed
saving in distance computations. A negative result rejects that broad premise,
not all possible layouts or GPU implementations.

Quake (OSDI 2025), https://www.usenix.org/system/files/osdi25-mohoney.pdf,
already uses partition sizes/access frequencies for maintenance and adaptive
query scanning. Ada-IVF, https://arxiv.org/abs/2411.00970, already addresses
partition imbalance/reconstruction error with selective local reindexing.
Generic workload-aware compaction/reclustering is not a new thesis. The parent
already establishes that ordinary dense fusion removes much survivor movement.
No new framework or expensive experiment is admitted by this CPU diagnostic.

## Fixed inputs and trace

Public SIFT1M archive from ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz.
Bind archive and extracted prefix hashes. Read first 262144 base rows, first
8192 learn rows, first 1000 query rows; 128D FP32 input, exact integer L2 ranking.
Train 64 centroids for 12 Lloyd iterations, deterministic seed 20261015; training
and assignments use FP64 and one BLAS thread. Query evaluation uses integer
squared L2 distances and (distance, original source row ID) tie order. Source
rows have unique IDs and are never reinserted, so no duplicated-vector generation.
Use 32768 and 65536 live rows; 64 epochs, 512 unique deletions and new insertions
per epoch. Each epoch commits before any snapshot query. Update workloads:

- random: uniform live deletion, uniformly shuffled unused source rows;
- FIFO: delete oldest live IDs, same unbiased insertion stream;
- cross-hot: uniform live deletion, insert from 16 fixed hot lists. Hot lists
  are the most populated 16 in the unused reservoir; selecting them is part
  of workload construction, not an indexing oracle.

Exhaustion or missing/duplicate IDs aborts; no workload is silently resized.
Check live-set/ownership conservation each epoch. At epochs 0/16/32/64, evaluate
64 original queries and 64 queries whose nearest centroid is hot (first such
queries in the 1000-query prefix). These are controlled traffic patterns, not
observed production query/update distributions.

## Same-contract representations

- **Fused Dense** keeper: final dense prefix per list, incoming rows first fill
  holes; only surviving rows outside that prefix relocate, matching the parent
  fused planner's count. No full-list rebuilding is charged as payload movement.
- **Reusable holes + prefilter**: per-list deleted slots can be reused by same-list
  insertions; trim trailing holes, never move surviving payloads. Check occupancy
  before reading a vector. This is an ordinary diagnostic representation, not a
  novel index, lock-free protocol or memory allocator.
- **Late validity control**: same hole state and results, but count a distance for
  every traversed slot before validity/top-k admission. It is a weak diagnostic
  only; never the primary performance comparator.

All use the same live IDs, payloads, fixed centroids and exact probed-list sets.
Per nprobe in 1/2/4/8/16/32/64, retain per-query IDs, Recall@10, live distances,
hole-slot metadata checks, scan-work quantiles and extent rows. Report minimum
listed nprobe meeting mean Recall@10 >= 0.90 / 0.95; do not assume equal nprobe
implies equal quality between different indexes. Here representations must have
identical selected-list answers and quality by construction AND explicit tests.
P99 of inspected rows is a workload-count quantile, not latency P99.

## Work and memory scope

Count payload row writes due to insertions and survivor relocation separately;
FP32 payload write bytes = rows * 128 * 4 (not read+write DRAM traffic).
Count live distance evaluations and occupancy entries inspected separately;
metadata counts cannot be compared to payload bytes as if their costs were equal.
Report logical extents, live rows, holes, maximum summed extents and per-list
high-water rows. These are representation bounds, not allocator capacity/VRAM.
Growth allocation/copy, host planning, ID lookup, routing, top-k, synchronization,
cache effects and hardware traffic remain unmeasured. Root/slot state is serial.

## Gates and decision

Before the full run: exhaustively test small fused plans against the parent
algorithm and a minimum-move count, test tail trimming/hole reuse, ensure an
injected invalid ID is rejected, compare matrix L2 ranks to direct integer L2,
and prove late-validity accounting cannot change returned live IDs.
Re-run from a saved commit. Preserve raw per-epoch and per-query evidence,
input/source hashes and exact software versions. No candidate is promoted from
counts alone. If prefiltering removes invalid distances, do not call historical
tombstones an inherent distance-computation cost. If metadata amplification
remains, quantify its range and reopen only for a concrete non-incremental GPU
mechanism and same-contract timing beyond ordinary filtering/maintenance.
