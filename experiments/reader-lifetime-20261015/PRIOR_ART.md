# Nearest-art kill test

Accessed 2026-10-15. Primary-source statements, not reproduced paper speedups.

- [Awad et al., PACT 2022](https://escholarship.org/uc/item/4mz5t5b7),
  Sections 3.2 and 4.4: host/on-stream/tile snapshot scopes; GPU DEBRA with
  block-wide announcements and shared-memory limbo bags. Execution scope is
  distinct from per-object or per-partition reachability. We directly use its
  [reclaimer](https://github.com/owensgroup/MVGpuBTree/blob/f755006d78ccca565f97cda053727d4d48b82d46/include/memory_reclaimer.hpp)
  unmodified, commit f755006d78ccca565f97cda053727d4d48b82d46. This is not a
  full-tree performance reproduction, and our direct-root snapshots are not
  its multiversion B-tree timestamp/version-chain algorithm.
- [Ben-David et al., DISC 2021](https://drops.dagstuhl.de/storage/00lipics/lipics-vol209-disc2021/LIPIcs.DISC.2021.12/LIPIcs.DISC.2021.12.pdf):
  the earliest active reader can retain unnecessary intermediate versions;
  range tracking and version-list removal address that existing problem.
  Identifying this phenomenon or retaining only snapshot-needed versions is
  not a new contribution. A GPU port alone would not establish novelty.
- [Brown, PODC 2015](https://mc.uwaterloo.ca/pubs/debra/paper.podc15.pdf): DEBRA and stalled-reader
  reclamation trade-offs predate this application. The GPU implementation
  explicitly cites this work.

Gate 0: do not invent a new system name or rescue a known lifetime mechanism
with a favorable injected delay. The diagnostic tests problem pressure only.

## Screen for the direction switch

- [GPU LSM](https://arxiv.org/abs/1707.05354): dynamic GPU dictionaries with
  batched updates and retrieval/update trade-offs. Base-plus-delta/merge alone
  is not a new GPU mechanism.
- [Jasper paper](https://arxiv.org/html/2601.07048v1) and
  [current source](https://github.com/saltsystemslab/Jasper/tree/7ec9125049d6ca08170e47f4027a10c7e37d18bc):
  GPU Vamana/RaBitQ, streaming updates; source has stable IDs, deletion flags,
  consolidation and physical compaction. Source and paper are distinct versions;
  these are not our reproduced performance results. Generic tombstones, batch
  repair or GPU quantization therefore do not qualify as the next thesis.
