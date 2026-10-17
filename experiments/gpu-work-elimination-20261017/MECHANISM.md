# Hypothesis: factor repeated query-mask work, not candidate selection

Frozen 2026-10-17 before the first count experiment. Parent `f198d6b8`.
GPU comparator source: `Stardust-SJF/cuvs_rabitq` at
`33d1de70fba2bd86390ea5915c62917c3a85c2ff`. CPU-only first stage. No GPU
initialization, polling, launch or performance claim; old CPU timings are unused.

## What would cease to be necessary?

The QUANT4 first filter performs four signed AND-popcounts for every candidate
word and query. If queries probing the same list have identical 32-bit query
masks, those **integer operations**, not just code loads, can be shared. This
holds even for candidates that the existing 1-bit filter later rejects.

For a candidate word `x`, canonicalize a query-plane mask with its complement:
`pc(x & ~m) = pc(x) - pc(x & m)`. Combine signed plane weights (1,2,4,-8)
per query and remove zero coefficients. Compute each remaining canonical mask
once for all queries sharing a list; reconstruct each original integer dot
product with the original per-query weights. No threshold, answer, FP32 payload
or quantization change is involved. No query may borrow future batches.

This changes the independent-query work organization, not the quantization
contract. It is **not** a claim that common-subexpression elimination or the
complement identity is new. The open research question is whether list-incidence
and mask redundancy support a non-incremental GPU execution mechanism after
charging reconstruction, dictionary construction and live state.

## Gate 0 and attribution controls

- The pinned GPU source already contains query-global atomic threshold updates,
  representative/centroid seeding, warmup reordering, candidate compaction,
  shared query preparation and adaptive IP granularity. Do not reinvent them.
- Ordinary batch tiling/cache reuse and bit-serial matrix products are known.
  Sharing code loads alone is not the hypothesis. See the source references in
  BOTTLENECK.md, [BISMO](https://arxiv.org/abs/1806.08862) and
  [RaBitQ](https://arxiv.org/abs/2405.12497).
- Progressive quantization already appears in
  [SAQ](https://arxiv.org/abs/2509.12086) and
  [bitplane-ann](https://github.com/olanokhin/bitplane-ann/tree/de9a37455f399e350691df963021c93b4458eb4b).
  This probe neither adds another quantization stage nor claims their contribution.
- Computation reuse from repeated quantized operands is also established in
  [UCNN](https://arxiv.org/abs/1804.06508) and
  [CREW](https://arxiv.org/abs/2107.09408). The generic reuse idea cannot be
  claimed as a new contribution; ANN-specific execution differentiation is open.
- Mandatory comparator: per-query mask canonicalization, not just native four
  operations. Cross-query gains must be incremental beyond that local control.
- A positive count result is not a novelty pass, GPU correctness or speed win.
  A nearest-art collision or uncompetitive net cost stops this implementation.

## Frozen cheap test

1. Reuse the previous SIFT/GIST/Gaussian inputs, centers, assignments and rotation.
   Use held-out queries 32..95 and the previous calibration's nprobe (16/16/64).
2. Add SIFT base prefix 262144 using the same centers, disjoint training rows and
   fixed nprobe16; explicitly report routing recall over this larger prefix.
3. Add public ANN-Benchmarks GloVe-100 angular: train rows0..32767 as base,
   rows32768..40959 as clustering train, test rows0..95 as queries. Normalize to
   L2 and pad to128. Query0..31 calibrates nprobe1/4/16/64 for recall .95;
   queries32..95 evaluate without retuning. This is word-semantic, not a modern
   sentence-embedding serving workload. The whole original GloVe corpus is not
   searched in this bounded screen.
4. CPU Faiss1.15.1, NumPy1.26.4, one BLAS/OpenMP thread, Xeon Platinum8575C.
   New clustering only for GloVe:64 centers,12 iterations,seed20261017.
5. Use the pinned source's signed4-bit query normalization/round/clamp/packing
   formula. The scaling-calibration algorithm uses100 PCG64(seed7) unit-normal
   samples in place of RAFT's RNG. Rotation is the preserved Faiss seed123 matrix,
   not a reproduced native GPU index. Retain scaling multipliers .9/1/1.1;
   never select the best from evaluation. Code/factor parity with a GPU-built
   index remains unvalidated. The exact factorization works for any input masks.
6. Batch1/8/32 are contiguous evaluation queries. Group only existing query/list
   incidences inside each batch. Compare native4R, per-query canonical CSE,
   batch-identical-mask CSE, and batch-complement CSE. The candidate chooses
   the cheaper of identical/complement bases per word using query masks only;
   ties use identical. The local control likewise chooses its cheaper basis
   independently per query/word. Complement factoring can add a whole-word
   popcount, so retain both raw ablations and any regression. Retain every group.
7. Weight primitive counts by actual list size. Native denominator is
   `list_rows * words * 4 * queries_in_group`. Count weighted reconstruction
   updates separately. Per-group descriptor byte model per word is
   `4*unique_masks + 2*nonzero_terms + 2*(R+1) + R + 8`; keys/slot IDs/int8
   coefficients/offsets/base coefficients/header. Add query-key input bytes and
   pair-grouping metadata. Scratch model:32 candidates times unique popcount
   outputs (uint8) plus32*R int32 accumulators. Not register allocation or VRAM.
8. Validate integer accumulators against dense bit/code dot products and a
   separate native four-plane loop, including negative codes, complementary
   masks, zero masks, all-ones masks and cancellation. Test real data words for
   every primary-scale batch32 group. No integer narrowing; reject invalid codes.
9. Global GT is exhaustive FP64 over each selected base prefix, stable IDs. The
   factorization preserves the prefilter's integer accumulator, not an unmeasured
   end-to-end Recall/QPS claim. Full GPU outputs and global recall remain open.

## Stop / next gate

If there is little incremental arithmetic elimination beyond local CSE, or
reconstruction/live-state costs consume the plausible saving, stop this path.
A large count signal warrants only a production-code parity and net-cost test
on an allocated GPU, not a paper. No hard work-count-to-speed conversion is
allowed. For any later GPU test keep the native comparator, identical index,
actual recall, total query preparation, sorting, warmup, filter, refinement and
merge in scope. Batch1 needs a keeper fallback; the CPU test does not validate it.

The source has a separate `k>64` path that computes all extended-code IPs.
Treat that as a comparator coverage/implementation issue, not a novelty win
obtained by changing one constant. This probe targets the `k<=64` filter path.
