# Quiescent invariants

- **I1:** Every reachable valid slot has a unique logical ID, an expected live record, and a matching ATT entry. Every expected live ID is reachable.
- **I2:** Active IDs have different physical coordinates; retained payload equals the deterministic CPU oracle.
- **I3:** Heads/next pointers stay inside the pool; each list is acyclic; slabs and records belong to exactly one list.
- **I4:** Reachable slabs and active free-stack entries are disjoint; the stack has valid, unique slab IDs and a valid top.
- **I5:** Each reachable slab's `valid_count` equals its bitmap population count after synchronization.
- **I6:** A repeated delete returns zero and leaves the already-quiescent state unchanged.
- **I7:** A quiescent full-list search returns the CPU exact L2 top-k, with `-1` padding where needed.
- **Resource conservation:** Every slab is reachable or free. An allocated, unreachable, non-free slab is an orphan. This is checked separately from the numbered record invariants.

Snapshots intentionally include every slab's metadata and every valid slot's ID/payload, all ATT entries, all heads, and the active free-stack prefix. Unused payload bytes are not logical records and are not serialized. Search evidence includes actual and reference labels/distances.
