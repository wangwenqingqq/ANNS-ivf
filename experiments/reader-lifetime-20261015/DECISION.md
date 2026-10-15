# Decision: no new reclamation thesis; keep the GPU question explicitly open

## Phenomenon

The CPU model reproduces the established retention mechanism: a protected reader
can stall native three-bag EBR progress, while the pool drains normally after
release. Initial GPU admission demonstrates protected reads overlapping real
acknowledgments, not a completed pressure/throughput comparison.

## Explanation and competing interpretation

Pre-read protection is avoidable under the declared root-load linearization
contract; protection through the final actual read is not. Moving acquisition
later is ordinary scoping, not an algorithmic invention. If invocation-time
snapshot semantics were required, the late variant would be inadmissible.
Our synthetic whole-partition COW may greatly amplify bytes relative to a
competent vector index; it is not an unavoidable ANN update cost. Shared race
reports and ineligible collection are not evidence that the established GPU
reclaimer is unsafe, slow or unable to support the real workload.

## Mechanism/thesis and implementation decisions are separate

- Novelty decision: **reject generic reader-lifetime/late-scoping/precise-MVGC as
  the next thesis**, on primary-source overlap, not on missing GPU measurements.
- Performance decision: **unvalidated**. No fair GPU advantage or negative
  advantage claim is admitted before current-source admission and the full
  matrix. Never use the early overbroad scope alone as a publishable baseline.
- Implementation checkpoint: keep the original comparator, failed records,
  current caller adapter and fail-closed resumption path. Do not repair upstream
  or suppress racecheck to manufacture an eligible baseline.

## Direction switch

Stop designing another epoch/reclamation protocol. The next problem screen is
**actual update-induced work amplification at the query/maintenance boundary**:
what payload/edge/quantization work is really necessary per successful churn,
and what query work grows before space is reclaimed? Use real vector replacements
and quality-matched competent indexes, not synthetic whole-partition COW alone.
This is a new measurement question, not a newly admitted mechanism.

Gate 0 already rejects two obvious repackagings:
- GPU base+delta/tombstones+periodic merge are established by GPU LSM and modern
  dynamic GPU indexes; do not relabel delayed deletion as innovation.
- [Jasper](https://github.com/saltsystemslab/Jasper/tree/7ec9125049d6ca08170e47f4027a10c7e37d18bc)
  already provides stable IDs, soft deletion, graph consolidation/compaction,
  GPU-native updates and RaBitQ. It is an additional external control, not a
  novel recipe to rediscover. Its API's synchronization/freshness semantics
  must be audited before any same-contract comparison.

Next cheap test after a device is available: finish the bounded original EBR
comparison only to close its evidence gap, then profile a pinned external
baseline's real replace/search/maintenance path with explicit successful IDs,
recall and physical/logical bytes. A new design requires a measured residual
and a non-incremental mechanism beyond those existing controls. No broad scaling
or paper drafting is authorized by a compilation result.
