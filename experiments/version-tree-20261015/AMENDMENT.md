# Descriptor accounting correction before primary_02

Date: 2026-10-15. This amendment follows initial observations and an independent
source audit; it is NOT retrospectively preregistered. Original CONTRACT.md,
primary_01 source/results and lookup receipts remain unchanged in ignored runs.

The initial history accounting reserved only sixteen eight-byte current handles.
An AVL search root is not generally the latest-version node, and array lookup
also requires a base/length descriptor. Initial block accounting charged units
but not current/held-root handles. Initial total-byte admission thresholds are
therefore incomplete, not admitted as equal-storage results.

Before primary_02, charge these explicit descriptor layouts:

- History chain: sixteen current/head handles = 128 bytes.
- History AVL: sixteen current handles + sixteen search-root handles = 256 bytes.
- History array: sixteen current handles + sixteen base/length/capacity
  descriptors (24 bytes each) = 512 bytes.
- Block layouts: one current root and eight reserved held-root slots = 72 bytes,
  charged from initialization, including before readers are registered.

Keep every original shape, policy, sequence, byte limit, entry/node size, seed
and correctness gate. Rerun all cases without selecting favorable outcomes;
retained-history lookup receipts and per-success block allocation counts must
remain byte-identical/logically unchanged. Preserve both failed-admission and
snapshot visibility checks. All storage bytes must equal units/entries PLUS
the declared descriptors, including high-water and post-release states.

This is still an explicit storage-layout model, not total production memory:
validation token/oracle containers, allocator/free-list metadata, collector
worklists, locks and implementation-specific alignment/capacity rounding are
not modeled. Concurrent reclamation is not implemented. New-unit byte counts
exclude in-place current/pin-handle writes; do not label them total update writes
or hardware traffic. Descriptor reservations do not establish a complete GPU
allocator/SMR memory budget. No result is a latency, novelty or GPU claim.
