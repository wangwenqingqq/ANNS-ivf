# CPU follow-up: topology is not reclamation

**Under the fixed held-snapshot trace, adding back links does not change DEBR
pool pressure. Precise collection changes it equally for singly- and doubly-linked
versions.** This is a CPU state-model result, not GPU performance or a new system.
No GPU execution or availability polling was performed for this follow-up.

## Frozen comparison and results

16 current roots; eight readers retain timestamp zero on roots 0..7; 512
round-robin replacement attempts. Both topologies perform the same batched
classification pass and logical pruning; the singly-linked control remembers
its predecessor instead of repeatedly searching from the head. The main trace
has held snapshots but no paused traversal cursor.

| Retention policy | Topology | Reserved payload slots | Acknowledged / 512 | First failed attempt (zero-based) | Peak physical slots | Linked / resident before release | Final roots after drain |
|---|---|---:|---:|---:|---:|---:|---:|
| Native DEBR transition model | singly | 64 | 48 | 48 | 64 | 24 / 64 | 16 |
| Same DEBR | doubly | 64 | 48 | 48 | 64 | 24 / 64 | 16 |
| Oracle-assisted precise GC | singly | 64 | 512 | none | 25 | 24 / 24 | 16 |
| Same precise GC | doubly | 64 | 512 | none | 25 | 24 / 24 | 16 |
| Native DEBR transition model | singly | 256 | 240 | 240 | 256 | 24 / 256 | 16 |
| Same DEBR | doubly | 256 | 240 | 240 | 256 | 24 / 256 | 16 |
| Oracle-assisted precise GC | singly | 256 | 512 | none | 25 | 24 / 24 | 16 |
| Same precise GC | doubly | 256 | 512 | none | 25 | 24 / 24 | 16 |

In the 64-slot DEBR case, 40 unlinked versions remain physically resident while
the old readers hold protection; at 256 slots this becomes 232. The topology
ablation has identical per-attempt occupancy and acknowledgment traces. The
precise trace's peak of 25 includes an in-flight replacement; its post-GC steady
state has 16 current plus eight snapshot-needed versions. This is not a bound
with arbitrary paused cursors, a query/second result, or a bytes-budget win.
Doubly-linked storage needs one additional back-link field per reserved slot;
Python object allocation is not used to estimate an actual GPU layout.

## Two counterexamples that the basic idea must handle

### Neighboring removals require a protocol

Of all 70 schedules of the declared two naive four-step removals of B and C in
A-B-C-D, **56** leave invalid surviving endpoint links. Both serialized orders
are valid. One retained witness is `BBBCBCCC`, where each letter executes the
next step of that node's removal: read previous, read next, write previous.next,
write next.previous. It ends with A.next=C and D.previous=B instead of A-D.
There was no freeing or weak memory in this test. These 56/70 schedules are
**not a probability or observed failure rate**, nor a finding against any
published concurrent doubly-linked-list algorithm.

### A directly protected cursor can retain an indirect transit path

A query needs v0 but pauses with a cursor at v2. The collector can shorten the
root-visible chain from v3-v2-v1-v0 to v3-v0 while that cursor still sees v2-v1-v0.
Freeing v1 because no snapshot selects it and no cursor directly announces it
creates a dangling traversal. The negative control updates free accounting so
that its rejection specifically reports `dangling version reference`, rather
than merely noticing a counter mismatch. The safe model retains the old cursor's
older-link reachability closure until progress/release. Both topologies pass.
This illustrates why precise visibility, structural unlink and safe physical
reclaim are separate tasks; it is not a complete hazard-pointer or reference-
counting implementation.

## Reproduce and verify

From the repository root, Python 3.9+ standard library only:

```bash
EXP=experiments/reader-lifetime-20261015
python3 "$EXP/model.py" --self-test
python3 "$EXP/cpu-unlink/check.py" --out /tmp/cpu-unlink-results.json
cmp /tmp/cpu-unlink-results.json "$EXP/cpu-unlink/RESULTS.json"
```

All eight per-attempt traces and all 70 removal schedules are in `RESULTS.json`.
The checker also runs 100 fixed seeds x 500 legal reader/update/cursor events per
topology, verifies snapshot answers, slot ownership, traversable older paths,
linked-chain back links and final drain, and rejects the controlled premature-free
mutation. Results were rerun
independently on the local and remote CPUs with byte-identical outputs.
`MANIFEST.json` binds the current source/contract/results and the preserved
initial run. The initial negative mutation only tripped generic accounting;
it was strengthened before this result to isolate the dangling cursor edge.
The initial bytes remain in ignored `runs/initial_01/` rather than overwritten.

## Scope, novelty and decision

The existing native-DEBR transition model is imported unchanged. The additional
version-chain model introduces explicit cursor traversals that the earlier GPU
direct-root probe does not contain. The precise collector is serialized,
oracle-assisted and uses unique logical generation handles. It does not prove
concurrent snapshot acquisition, unlink/RC correctness, ABA protection, CUDA
memory ordering, bounded metadata or GPU speed. Its scan/rewire counters are
logical model operations, not GPU instructions or timings.
Queries and cursors follow older links only; detached newer fields can be stale
and are never dereferenced. The checks do not establish bidirectional-reader
safety. This scope clarification was added after independently reproducing an
audit witness with a live older-only cursor and a stale detached back link.

[Ben-David et al., DISC 2021](https://drops.dagstuhl.de/storage/00lipics/lipics-vol209-disc2021/LIPIcs.DISC.2021.12/LIPIcs.DISC.2021.12.pdf)
already combines version eligibility, doubly-linked version-list removal and
safe memory reclamation. This experiment is an explanatory ablation, not a
reproduction of that concurrent algorithm or evidence for a new MVGC thesis.
Keep DLL as an engineering option, but reject 'add prev' as a standalone fix for
old-reader retention under the tested contract. A GPU advantage remains unknown.

| Claim | State | Evidence / allowed wording |
|---|---|---|
| Topology-only changes the fixed DEBR occupancy trace | rejected | Eight-case topology/retention ablation; S and D traces equal within each policy |
| Precise collection lowers payload-slot retention here | partial | Serialized oracle-assisted CPU model; identical S/D peaks, not a deployable protocol |
| Naive neighboring pointer edits always preserve structure | rejected | 56 invalid schedules among the exactly declared 70; not a failure probability |
| Direct cursor announcements alone cover every traversed reference | rejected | Retained v2-v1-v0 counterexample, cause-specific negative mutation |
| Bounded CPU checks pass | measured | All declared assertions, 100 seeds per topology, deterministic result hashes |
| A new or faster GPU mechanism is established | unknown | No GPU execution, no concurrency implementation or latency comparison |
