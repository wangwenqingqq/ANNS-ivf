# Reproducible issues

Source: upstream `c41c59d8f65e7a1eefac4aba500e8b6f17b6f546`.
The uninstrumented executable compiles the original append, search, and deletion
source files. All state assertions below are made after synchronization.
Raw paths are relative to the run named in `EVIDENCE_RUN.txt`.

## R1 — A partial hole causes an active tail slot to be overwritten

**Reproduced, unmodified algorithm.** One list, 32 serially inserted vectors,
D=8, pool=128. Read the physical slot IDs from snapshot A; delete slot 5's ID
(observed ID 5); insert ID 100.

- Before deletion, slab 127 is full and ID 31 occupies slot 31.
- After insertion, ID 31 and ID 100 both have ATT coordinate `545460846623`
  (`slab=127, slot=31`). Physical slot 31 contains ID 100; ID 31 is missing.
- `valid_count=32` while the bitmap has only 31 set bits. Querying ID 31's
  original vector no longer returns the CPU exact top-k.
- The matched slot-31 deletion control passes the state and search checks.
- Repeating either deletion returns zero and preserves the snapshot: the tested
  quiescent deletion-idempotence control passes.

Root boundary: append uses the pre-increment `valid_count` as a physical slot
index; deletion decreases that same live count without compacting active slots.

Evidence: `snapshots/T1_hole/{A,B,B_repeat,C}.json`,
`snapshots/T1_hole/search.json`, matched `T1_tail` files, and `logs/repro_T1_*.log`.

## R2 — Recycling an empty slab leaves a live list reference

**Reproduced, unmodified algorithm.** Two lists, D=8, pool=128. Construct 64
vectors (head test) or 96 vectors (middle test) with serial single-vector
insertions; delete all observed IDs in slab 126.

- The emptied slab remains reachable and simultaneously appears in the free
  stack: I4 fails before any reuse.
- Inserting ID 100 into list 1 reuses slab 126, still referenced by list 0.
  Both lists now reach the same slab and list 0 can return list 1's ID.
- Reinitializing that slab's `next` disconnects the old tail: 32 expected live
  IDs are no longer reachable, and one slab becomes an orphan.
- Both head and middle cases fail; their pre-deletion snapshots pass.

Root boundary: deletion publishes the slab into the free stack without changing
the list head or predecessor link. Reuse then mutates metadata still owned by
the old chain.

Evidence: `snapshots/T2_head/`, `snapshots/T2_middle/`, and corresponding logs.

## R3 — Failed head CAS attempts permanently consume pool capacity

**Reproduced naturally, without delay injection.** One hotspot list, D=8;
1000 independently reset trials per batch, with a pool 64 times the batch.
All attempted IDs commit in this ample-pool matrix, but every trial leaks slabs.

| Batch | Trials with orphans | Orphans per uninstrumented trial | Minimum free slabs | Minimum committed IDs |
|---:|---:|---:|---:|---:|
| 64 | 1000/1000 | 94 | 4000 | 64 |
| 256 | 1000/1000 | 924–1103 | 15273 | 256 |
| 1024 | 1000/1000 | 16376–16727 | 48777 | 1024 |

In a separate 1000-batch accumulating run (64 vectors/batch), the final state is
64000 expected/reachable IDs, 2000 reachable slabs, 35072 free slabs, and **94000
orphans** in a 131072-slab pool. This is a loss of reusable preallocated pool
capacity, not a measured increase in `cudaMalloc` bytes.

A separate count-only build confirms, in every recorded trial, that orphan
count equals failed head-CAS count, allocated slabs equal reachable plus orphan
slabs, and append reclamations are zero. Instrumented scheduling/count ranges
are not substituted for the original run. The initial smaller-pool pilot is
retained separately; its exhaustion-limited results are not the ample-pool table.

Evidence: `snapshots/T3/trials.csv`, `snapshots/T3_counted/trials.csv`,
`snapshots/T3_growth/growth.csv`, first/final snapshots, and corresponding logs.

## R4 — Failed insertion is not propagated through the public API

**Reproduced under controlled free-stack exhaustion.** The raw one-slab test
fills all 32 slots and attempts ID 100: 33 attempts, 32 committed IDs.

The real public-index API control first inserts 32 records successfully. Its
allocator auto-expands to 4116 slabs; the test then deliberately sets its free
stack top to zero while keeping the current head full. This is an injected
unavailable-pool condition, not a naturally exhausted large public index.
`add_with_ids` returns without an exception; ID 100 retains an invalid ATT;
`ntotal` nevertheless increases from **32 to 33**.

Only a test-derived class exposes existing device state; the upstream API and
insertion algorithm are not patched. This establishes failed-insert reporting,
not a claim about the natural frequency of public-index pool exhaustion.

Evidence: `snapshots/T7_pool/`, `logs/repro_T7_pool.log`,
`logs/repro_T7_public.log`, and `tests/test_public_api.cu`.

## Supplement — Free-stack publication order

The unmodified allocator-primitive test did not return a stale slab in 1000
natural trials. A separate copied-header build pauses after publishing the
increased stack top and before writing the released slab ID. It returned the
stale, still-held slab in **9/10** trials; one trial did not expose the stale
read. The pause is bounded to avoid a hung GPU, and can perturb scheduling.

This is **fault-injection evidence for allocator primitives**, not a natural-run
failure of the complete live-index Insert/Delete path. Do not promote it to
that stronger claim. Full-index publication interleavings remain unvalidated.
