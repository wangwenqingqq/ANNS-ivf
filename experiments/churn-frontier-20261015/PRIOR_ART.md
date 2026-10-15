# Gate 0: nearest work and cheap novelty kill tests

Accessed 2026-10-15. Facts below are statements from primary papers/docs, not
independently reproduced performance results. The search is a scoped screen,
not a proof that no other prior work exists.

| Work | Established mechanism / closest overlap | Decision for this round |
|---|---|---|
| [ETALE](https://arxiv.org/html/2607.02543v1), 2026 | GPU graph: immutable adjacency slabs, packed atomic COW publication, retired/free separation with kernel-boundary reuse | Kill “safe COW slabs + epochs” as the thesis; do not transfer its graph claims to IVF or assert its proof is validated |
| [GrAND](https://arxiv.org/html/2608.21163v1), 2026 | GPU graph: consolidate affected-node updates and repair across a batch; find/replace and on-demand reverse graph | Kill “group updates to remove conflicts” as a standalone contribution |
| [Quake](https://www.usenix.org/system/files/osdi25-mohoney.pdf), OSDI 2025; [code](https://github.com/marius-team/quake) | Adaptive vector-index partitions and maintenance driven by the query/update workload | Kill generic workload-aware repartition/maintenance; any GPU-specific residual needs a different mechanism and same-contract evidence |
| [Ada-IVF](https://arxiv.org/abs/2411.00970), 2024 | Select problematic IVF partitions for local maintenance/reclustering | Do not call selective local maintenance new |
| [RTAMS-GANNS](https://arxiv.org/abs/2408.02937), 2024 | Multi-stream GPU online insert/search execution | Merely overlapping CUDA streams is not novelty |
| [SVFusion](https://arxiv.org/abs/2601.08528), 2026 | CPU/GPU/disk hierarchy, caching and coordinated query/update execution | “Dynamic search on GPUs” is already too broad as a contribution claim |
| [cuVS IVF-Flat](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/ivf-flat) and [source](https://github.com/NVIDIA/cuvs) | Coarse-list scanning, original vectors in index lists, list capacity/allocation trade-offs and incremental extend | A future quality/QPS comparator; this diagnostic is not a cuVS replacement or measurement |
| [GPMA+](https://desert0616.github.io/publications/vldb_gpma.pdf) | Sort/group updates by segment and use GPU primitives | Batch ownership and conflict removal are established beyond ANN |
| [Fully Concurrent GPU Data Structures](https://escholarship.org/uc/item/5kc834wm), 2022, Chapter 4 | GPU multiversion B-tree, scoped snapshots and safe memory reclamation | Generic GPU MVCC/scoped epochs are also not a new thesis; any reclamation proposal needs this comparison |
| [faimGraph](https://hpdgp.mpi-inf.mpg.de/pappe/faimGraph.pdf), SC 2018 | Fully dynamic GPU graph storage and memory-management queues | Do not confuse dynamic GPU storage with a new ANN contribution |
| [MARS](https://www.fratepietro.com/papers/MARS/main.pdf), 2026 | Episode-scoped GPU retrieval; deletion discussion uses tombstones with periodic compaction and leaves formal eviction policy open | Adjacent temporal-scope work, not evidence that bounded arbitrary churn/reclamation is solved |

## Decision

No novel system is admitted at Gate 0. In particular, indirection, time windows,
ordinary fusion, bitmap allocation and epochs are not contributions by renaming.
Do **not** rescue these subsumed theses with additional engineering or a small
kernel gain.

One cheap diagnostic is justified to identify the *remaining problem*, not to
promote an existing technique: quantify the payload-mobility/query-locality
trade-off against a fused correct dense control under FIFO versus random/hot
cross-partition churn. If ordinary fusion removes the alleged bottleneck, retain
that negative evidence and switch questions. If it does not, the next direction
still needs a non-incremental mechanism and an expanded nearest-art screen.
