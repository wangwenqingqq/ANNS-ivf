# CPU-only unlink/retention ablation

Date: 2026-10-15. Experiment: ivf_20261015_unlink_retention_sc_model.
Parent: 10d92b7f2fd52cb070a63ec9d98a92879ec13a16. State: preregistered.
The user selected CPU-only continuation until they identify an available GPU.
No GPU execution, device polling, or power/process changes in this experiment.

## Questions and attribution

1. Does a back link by itself remove finite-pool pressure under an old reader?
2. Which change, topology or eligibility/reclamation, changes resident versions?
3. Can naive adjacent removals corrupt a doubly-linked list?
4. Is protecting only directly held cursors enough after intermediate unlinking?

This is a sequentially consistent state model, not a concurrent implementation,
CUDA-memory-order proof, a GPU result, or new MVGC. The nearest control is the
range-tracking/version-list/reclamation separation in Ben-David et al., DISC
2021 (see the parent PRIOR_ART.md). Our precise collector is serialized and
oracle-assisted; no claim of reproducing that paper's concurrent algorithm,
asymptotic guarantees, or its implementation is permitted.

## Fixed 2x2 component comparison

Sixteen current roots; eight active readers hold timestamp zero on roots 0..7.
512 round-robin replacement attempts; pools of 64 or 256 physical version slots.
The observable query is the latest version at its held timestamp on its root.
Each reader must retain the same result throughout the trace; successful updates
must be visible to newly invoked readers. OOM is explicit, not a successful write.

Topology: singly-linked versus doubly-linked version lists. Both use one batch
classification pass per root for GC: the singly-linked control retains its
predecessor during that pass, rather than rescanning the list for each victim.
This avoids an artificial quadratic baseline. Doubly-linked removal can use a
known newer neighbor but pays additional back-link storage/writes.

Retention: the existing native-DEBR transition Model (imported unchanged from
../model.py) versus a serialized precise visibility/reachability collector.
For this ablation, the DEBR case holds all eight initial readers throughout all
attempts, so retired addresses are not reused until final drain. Logical pruning
is allowed in BOTH policies; only physical-reclaim eligibility differs.
The precise case can release an unlinked version only if it is not reachable
from a held traversal cursor. Such cursors retain their older-link closure;
protecting only the directly announced cursor would be unsafe.
Readers and cursors traverse older links only. Detached newer fields may become
stale and are never dereferenced; back-link consistency is required only within
linked root chains. This is not a safety check for bidirectional readers.

Report acknowledged/failed attempts, first failure, physical slot high-water,
linked versus unlinked-resident versions, and final drain to 16. Equal payload
slot capacities are not equal total-byte budgets: a doubly-linked reservation
has one additional back-link field per slot. No CPU or GPU speed ratio is reported.
Oracle histories and unique logical generation handles are validation machinery,
not a bounded-memory implementation or an ABA solution.

## Bounded adversarial checks

- Enumerate all 70 schedules of two naive four-step neighboring removals in
  A-B-C-D: read previous, read next, write previous.next, write next.previous.
  Delay freeing until both finish, so structural corruption does not depend on
  an allocator or weak memory. Keep both serialized orders as positive controls.
- Retain a snapshot of v0 and a traversal cursor at v2 in v3-v2-v1-v0. Unlink
  intermediate versions from the root. A direct-cursor-only free must be rejected
  if v2 still leads to v1. A reachability-aware control retains that transit path
  until the cursor advances/releases. This is a controlled negative mutation.
- Run 100 fixed random seeds, 500 legal start/update/cursor-step/end events per
  topology on the precise model. Check query visibility, dereferenceable older
  paths and linked-chain back links, unique slot ownership, explicit failure
  accounting and post-reader drain.

Primary decision: if topology-only ablations preserve DEBR resident pressure
while precise SLL/DLL have equal payload slot bounds, reject 'add prev to solve
retention' under this model. This does not reject DLL as an implementation
component, prove all SLL implementations equivalent, or establish a GPU win.
The GPU race/admission/full-matrix gate from the parent remains pending.
