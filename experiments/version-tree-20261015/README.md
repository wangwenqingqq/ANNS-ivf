# Replacing a version chain with a tree: CPU screen

**AVL improves long-history lookup over a descending chain, but the append-array
binary-search control matches its inspected-entry count at 4096 versions.
Topology does not remove the held-reader EBR retention in this trace.**
A separate persistent block-tree model reduces directory copy counts at large
block counts, but its protected residency is not uniformly lower. These are
ordinary known mechanisms, not a new GPU index or a measured GPU advantage.

Experiment: `ivf_20261015_version_tree_sc`, parent
`a139d04c17ea8dbf9cbef65f3f3cb5ffa812cad3`. [CONTRACT.md](CONTRACT.md) was frozen
before implementation; [AMENDMENT.md](AMENDMENT.md) records a post-observation
descriptor-accounting correction before the final rerun. This experiment is independent of other unpublished
experiments and changes no production source or earlier evidence.

## A. Timestamp-tree results

The histories are built online in increasing timestamp order. At each size,
all three controls answer the same 1030 predecessor queries: six boundary/age
queries plus 1024 fixed uniform samples, verified against Python's bisect oracle.
The table counts inspected entries, not latency, instructions, cache misses or
memory transactions. Histories are deliberately retained for these queries;
this is NOT the current direct-root GPU probe or the pressure workload below.

| Retained versions | Chain mean / P95 | AVL mean / P95 | Append array mean / P95 |
|---:|---:|---:|---:|
| 16 | 8.411 / 16 | 4.150 / 5 | 4.081 / 5 |
| 256 | 127.317 / 244 | 8.012 / 8 | 8.005 / 8 |
| 4096 | 2030.217 / 3890 | 12.003 / 12 | 12.003 / 12 |

At 4096 versions, AVL's online build performs 4083 rotations. Array append,
allocation/capacity growth, GC compaction and AVL mutation writes are not timed
or compared as a full update operation. These results therefore do not establish
array dominance for arbitrary updates. The direct latest-version handle avoids
history lookup in every control; adding a tree does not improve that zero-lookup
path merely because old-history searches are cheaper.

### Same retention policy, topology isolated

Sixteen roots; eight readers hold timestamp zero on roots 0..7; 512 round-robin
replacement attempts. The native DEBR transition model is imported unchanged
from the parent experiment. The precise control is a serialized, oracle-assisted
collector, not a concurrent reclaim protocol. Lookups and index edits are atomic
model operations: no paused traversal or concurrent rotation safety is tested.

| Capacity contract | Policy | Chain acknowledgments | AVL acknowledgments | Array acknowledgments |
|---|---|---:|---:|---:|
| 64 entries, not equal bytes | native DEBR model | 48 | 48 | 48 |
| 64 KiB declared layout budget | native DEBR model | 107 | 104 | 107 |
| 256 KiB declared layout budget | native DEBR model | 480 | 465 | 479 |
| Each of the above | precise model | 512 | 512 | 512 |

All precise cases peak at 25 versions, including an allocated replacement before
collection; all cases drain to 16 current versions. Native-DEBR pressure is not
removed by the search index. With equal declared bytes, the extra AVL metadata
instead reduces the admitted version count. This depends on the declared packed
fields, not actual Python object storage or a demonstrated GPU allocator layout.
No snapshot acquisition race or safe concurrent unlink algorithm is implemented.
Descriptor reservations are 128/256/512 bytes for chain/AVL/array respectively:
all have current handles; AVL adds search roots, array adds base/length/capacity.
These are explicit storage fields, not a total production implementation budget;
allocator/free-list, GC workspaces, locks and alignment/capacity rounding are
not modeled. Oracle containers and Python object sizes are not charged.

## B. A different benefit: persistent block storage

This changes *payload storage*, not the timestamp lookup index. Each block
represents eight FP32 128D vectors (4096 payload bytes). Values in the executable
are integer generation tokens: no vector payload bytes, ANN queries or distance
computations are executed. The counts below charge each allocation its declared
packed-layout size; they are not measured CPU/GPU memory traffic or bytes
actually copied by the Python runtime.

The primary comparison copies one leaf plus an immutable **flat directory**
versus one leaf plus a persistent **binary-tree path**. Whole-partition COW is
retained only as a weak diagnostic, not the primary benchmark or an inevitable
IVF cost. Tree nodes have no metric bounds and do not skip live vector blocks.

| Blocks per partition | Flat directory: new-unit bytes / success | Block tree: new-unit bytes / success | Tree directory units in a full scan | Flat directory units in a full scan |
|---:|---:|---:|---:|---:|
| 16 | 4256 | 4240 | 15 | 1 |
| 256 | 6176 | 4368 | 255 | 1 |
| 4096 | 36896 | 4496 | 4095 | 1 |

The common leaf write is 4096 payload + 16 header bytes. Tree path copies add
32 bytes per level; flat directory copies add 16 + 8 x block-count bytes. At
4096 blocks this lowers new-unit byte counts about 8.21x relative to this flat
directory, but a full tree scan visits 4095 internal nodes (131040 bytes), versus
one 32784-byte flat directory. Directory-unit counts are not cache transactions;
neither copy-count nor scan-count ratios are GPU latency ratios. In-place
current/pin-handle writes are outside the new-unit byte count, which is not total
update writes. All block layouts separately reserve 72 descriptor bytes for one
current root and eight held-root slots, charged even before readers register.

### Protected residency and finite-budget counterevidence

Each case attempts 256 replacements with eight staggered held root snapshots,
under uniform and repeated-hot-block patterns. A common byte limit equals four
times the initial dense payload. Initial metadata and complete pre-publication
write sets are included. Failed admission is explicit and does not mutate root,
allocator serial or contents; no unacknowledged update is counted as a success.

- At 256 blocks, uniform updates under precise reachability complete all 256
  writes in both controls, with identical final logical contents and snapshot
  schedule. Tree residency before reader release is **1883240 bytes**, slightly
  worse than flat's **1877272 bytes**, despite lower copy counts.
- At 16 blocks under the same precise/uniform pattern, flat acknowledges **61**
  attempts and tree **60**. Their accepted histories differ after OOM, so their
  final residency/content hashes are not a same-history footprint comparison.
- At 4096 blocks, precise/uniform controls both acknowledge 256. Resident bytes
  before release are **18059672** (tree) and **18178216** (flat): a much smaller
  footprint difference than the per-success copy-count ratio.
- Every case drains to its current reachable graph plus the fixed descriptor
  reservation after reader release.
  `hold_no_reclaim` models only the no-free boundary during a stalled epoch;
  unlike Part A it does not execute the native DEBR transition implementation.
  It is not a reclamation-latency or epoch-overhead comparison.

Root/reference reachability, not just current values, drives collection in this
model. The model does not implement concurrent root acquisition, hazard/RC
updates, ABA protection, tree splitting, vector insertion/deletion, occupancy
filtering, allocator growth, or a GPU-efficient high-fanout directory. Ordinary
hole reuse, undo logs and other versioned blocked arrays are not eliminated as
competitors by this limited comparison.

## Validation and evidence

- All lookup receipts, all 18 x 512 pressure steps and all 36 x 256 block-storage
  attempts are retained in `LOOKUP.jsonl` and `RESULTS.json`.
- 100 seeds x 500 mixed index events per topology, including predecessor oracle,
  ordering, unique keys, AVL height and balance checks. A removed snapshot-needed
  version is rejected by the cause-specific visibility check in each control.
- 20 seeds x 64 mixed snapshot/update/release events per block layout and policy;
  sampled main-matrix reads every step, exhaustive retained-root values at the
  end, reference closure, byte accounting and exact drain checks.
- Complete-write-set admission rejects a no-space update without root or
  allocator/content mutation. The runner refuses to overwrite output directories.
- Primary CPU: Python 3.12.3; independent CPU reproduction: Python 3.9.6. No
  third-party dependencies or GPU initialization. Both deterministic outputs
  matched byte-for-byte. Delivery additionally requires a committed-tree release
  proof and independent publication verification.

The initial `primary_01` run omitted separate index descriptors and root/pin
tables; its lookup answers and new-unit counts are retained, but its incomplete
total-byte thresholds are not promoted. The amendment fixes these fields in
`primary_02` without changing shapes, policies, seeds, queries or unit sizes.
All original lookup receipts and block per-success counts remain unchanged.
`MANIFEST.json` binds published bytes and both preserved source/result stages.
Runs and machine-specific logs remain outside Git. The hash of the unchanged
parent DEBR model is in `RESULTS.json`; all generation tokens/oracles are
validation machinery, not a bounded-metadata production implementation.

## Decision and claim ledger

[MVGpuBTree](https://github.com/owensgroup/MVGpuBTree/tree/f755006d78ccca565f97cda053727d4d48b82d46)
already supports GPU snapshots/EBR. [DISC 2021 MVGC](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DISC.2021.12)
already addresses precise version retention. Balanced lookup and persistent
path copying are known techniques, not a new thesis. No GTS source is copied;
its distance-tree pruning is a different axis not exercised here.

| Claim | State | Allowed scope / counterevidence |
|---|---|---|
| AVL lowers old-history entry visits vs chain | partial | Fixed retained-history CPU queries; array matches mean at 4096, latest handles need no lookup |
| Replacing chain with AVL removes EBR pressure | rejected | Same 64-entry trace: all stop at 48; equal-byte AVL fills earlier under these layout assumptions |
| Block path copying can lower copied metadata vs flat COW directory | partial | Declared immutable layout, not actual hardware traffic, concurrent implementation or novelty |
| Block tree always lowers protected residency / increases acknowledgments | rejected | Matched 256-block precise/uniform residency is worse; 16-block finite-budget ack is lower |
| Bounded CPU checks establish a new/faster GPU index | unknown | No concurrent, CUDA, service, ANN-quality or latency gates run |

**Implementation:** executable CPU diagnostic only. **Mechanism:** retain
conventional block path copying as an engineering option; timestamp-tree search
does not fix the demonstrated retention boundary. **Thesis:** no new mechanism
or end-to-end advantage admitted. Do not implement CUDA solely to promote this
known combination. Reopen only for a concrete workload bottleneck, a mechanism
beyond competent existing controls, and same-contract evidence. Missing GPU
evidence is uncertainty, not proof against every possible tree design.

## Reproduce

From the repository root, Python 3.9+ standard library:

```sh
EXP=experiments/version-tree-20261015
python3 "$EXP/check.py" --self-test
python3 "$EXP/check.py" --out /tmp/version-tree-fresh
cmp /tmp/version-tree-fresh/RESULTS.json "$EXP/RESULTS.json"
cmp /tmp/version-tree-fresh/LOOKUP.jsonl "$EXP/LOOKUP.jsonl"
```

Use a fresh output directory. Lookup receipt columns are `[history_size, index,
timestamp, selected_version, inspected_entries]`; pressure traces are `[attempt,
acknowledged, resident_versions, declared_resident_bytes]`; block traces are
`[attempt, logical_key, value_token, acknowledged, new_bytes, freed_bytes,
resident_bytes]`. Attempt indices are zero-based. All GPU gates from the parent
remain unchanged. No GPU was launched or polled for this experiment.
