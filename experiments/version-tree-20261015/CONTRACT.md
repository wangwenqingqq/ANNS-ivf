# CPU version-tree screen

Experiment: ivf_20261015_version_tree_sc. Date: 2026-10-15.
Frozen before implementation. Parent: a139d04c17ea8dbf9cbef65f3f3cb5ffa812cad3.
CPU only, Python 3.9+ standard library; no GPU execution or device polling.
This is a sequentially consistent executable model, not a concurrent index,
CUDA protocol, CPU timing benchmark, ANN-quality experiment or novelty claim.

## Gate 0 and hypotheses

[MVGpuBTree](https://github.com/owensgroup/MVGpuBTree/tree/f755006d78ccca565f97cda053727d4d48b82d46)
already supports GPU snapshots and EBR. [DISC 2021 MVGC](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DISC.2021.12)
already separates version eligibility, unlinking and safe reclamation. Balanced
predecessor indexes and persistent path copying are established mechanisms.
GTS distance-partition trees are not timestamp-ordered balanced trees. No GTS
code or third-party implementation is copied into this screen.

| Question | Keeper / candidate | Falsifier | Decision scope |
|---|---|---|---|
| Does tree lookup reduce long-history work? | descending chain / online AVL; append-array binary search is a competent extra control | no useful lookup reduction, or array matches it | inspected entries, not GPU speed |
| Does topology alone remove held-reader EBR pressure? | same native-DEBR transition model, three indexes | same payload retention, earlier OOM after metadata | fixed held-reader trace only |
| Does a block tree reduce update-copy work? | immutable flat block directory / persistent binary block tree; whole-payload COW is diagnostic only | flat directory is as cheap or tree overhead offsets benefit | explicit packed-layout model only |

Stop after the bounded matrix, correctness/mutation checks and independent CPU
reproduction. Do not implement CUDA, scale to a service, or draft a new thesis.
A known mechanism's count advantage does not pass novelty Gate 0.

## A. Timestamp history

1. Build histories online by increasing timestamps, of length 16/256/4096.
   Compare descending singly-linked records, AVL predecessor search and a
   sorted append array with binary search. Query timestamps include before-first,
   oldest, middle, newest, gaps, and 1024 deterministic uniform samples (seed 17).
   Check answers against an independent bisect oracle. Retain all query receipts.
   Count inspected entries and AVL rotations; no wall-time or hardware-byte ratio.
   Latest-version direct handles require no history lookup in all three controls.
2. Pressure: 16 current roots, eight held timestamp-zero readers on roots 0..7,
   512 round-robin replacement attempts. Use the parent model.py unchanged for
   native-DEBR transitions. The precise control is an oracle-assisted serialized
   collector retaining current and snapshot-selected versions. No paused lookup
   cursors or concurrent rotations/deletions are supported in this experiment.
3. Packed-layout accounting: 512 payload bytes per version; timestamp+link or
   timestamp+array entry = 16 metadata bytes; AVL timestamp+two children+height
   and padding = 32 bytes. All reserve 16 eight-byte current handles. Compare
   a common 64-entry capacity and equal 64/256 KiB byte budgets; the common-entry
   case is explicitly not an equal-byte comparison. Allocator headers, array
   capacity growth and free-list overhead are excluded; these are layout
   assumptions, not Python object sizes or measured VRAM. All OOM attempts fail
   explicitly and leave logical contents unchanged. Record peak including the
   allocated replacement before collection and the drain after reader release.
4. Validate all index invariants and snapshot answers every step; 100 seeds x
   500 mixed insert/delete/predecessor events per index. Remove a snapshot-needed
   key in a negative mutation and require a cause-specific visibility rejection.

## B. Block storage: a separate experiment, not a timestamp-tree benefit

One fixed-size partition contains 16/256/4096 blocks. Each block represents eight
FP32 128D vectors (4096 payload bytes); logical values are integer generation
tokens, not vector data or distance computations. Test 256 attempted point
replacements with seed 23 under uniform and repeated-hot-block patterns.
Capture eight root snapshots before attempts 0/8/16/24/32/40/48/56 and hold them
until the end. Query current and held states against immutable tuple oracles.

- Whole-payload COW: payload plus 16-byte header per version. Weak diagnostic.
- Flat directory: 16-byte header + eight-byte block handles; each leaf has
  4096 payload bytes and a 16-byte header. Copy one leaf and the directory.
- Binary block tree: same leaves, 32-byte internal nodes; immutable path copying.
  No distance pruning, rebalancing, insertion/deletion, occupancy change or split.

All use a common byte limit of four times the initial dense payload; include
initial metadata and all allocated units. Admission reserves the complete write
set before allocation/publication; rejected writes leave root and allocator
state unchanged. Report each attempt, acknowledgments, bytes newly written,
metadata/payload split, peak/resident bytes, and directory units visited by a
full scan. Compare per-success copy counts, not completion-time ratios or total
writes across unequal acknowledgment counts. Retain both hot/uniform cases.

Reclamation controls: (a) no freeing during the held-reader interval, representing
only the eligibility boundary of a stalled epoch, NOT a new generic-DEBR port;
(b) serialized reachability from current and held roots. After release, both
drain to exactly the current reachable graph. Check sampled values every step
and all values in all retained roots at completion. Include 20 seeds x 64
mixed snapshot/update/release events per layout with a large budget, and an
explicit failed-admission/no-mutation check.

This does not compare against every competent versioned flat array, undo-log,
high-fanout tree or GPU-native layout. In particular, ordinary hole reuse already
avoids survivor moves outside a COW contract. A block-tree result must not be
used to assert unavoidable whole-partition copying in IVF.

## Evidence and promotion boundary

The runner refuses to overwrite an output directory. Preserve failed/incomplete
runs and source hashes outside Git; publish source, contract, deterministic raw
receipts, manifest and a calibrated result ledger. Independently reproduce on
two CPU runtimes and from the committed tree. GPU correctness, weak-memory,
paused-traversal safety, ABA, reclamation latency, sustained throughput and ANN
quality remain unknown even if every bounded assertion passes.
