# Candidate-information frontier: first CPU opportunity screen

Date: 2026-10-15. Experiment: ivf_20261015_candidate_information_cpu_v1.
Frozen before measurement. Parent: c98871ad4217b3b11a68461d088d89d784481891.
This is the first bounded stage of the supplied open research brief, not a
completed GPU campaign or a claim that the full acceptance criteria are met.
No GPU launch, library/device initialization, process/power change or GPU polling.
CPU libraries only; one BLAS/OpenMP thread. Preserve every incomplete/failed run.

## Gate 0

Partial-distance pruning is established; ADSampling adds adaptive probabilistic
distance decisions. RaBitQ, extended RaBitQ and TurboQuant already address
low-bit information. GPU IVF-RaBitQ already combines 1-bit filtering, extra-code
refinement and fused execution, and can avoid raw-vector reranking. DARTH is
query-level adaptive termination, not the same candidate-level certificate.
None of these known recipes is a new contribution here. The current question
is opportunity AFTER accounting for evidence, threshold acquisition and execution
granularity; not whether IVFFlat alone can be beaten by ordinary quantization.

Primary references and source pins will be recorded in PRIOR_ART.md. Never
substitute a homemade sign sketch for the mature RaBitQ comparator or equate
CPU timing to GPU timing. GPU performance, warp efficiency and physical traffic
remain unknown until an allocated device permits a separately frozen campaign.

## Inputs and quality

- SIFT: first 32768 base vectors, FP32 128D; training base rows 262144..270335.
- GIST: first 16384 base vectors, FP32 960D; training base rows 16384..24575.
- For each real dataset, query rows 0..31 are calibration, 32..95 evaluation.
  Original query files, not self-queries. Preserve raw-prefix and decoded-array
  hashes. Verify public-mirror identity when available; explicitly record any
  provenance limitation instead of inferring identity from filenames.
- A labeled weak-structure diagnostic: 8192 independent 960D normalized Gaussian
  base vectors, 8192 training, 96 query vectors (PCG64 seed 20261015). This is
  synthetic negative-control evidence, not a replacement for real queries.
- Ground truth is exhaustive FP64 squared L2 over the ENTIRE selected base
  prefix, with stable (distance, source-ID) ties. It is not full-million GT.
  Return global Recall@10, per-query results, and selected-candidate exactness.
  Check the distance calculation against independent direct subtraction.

## Fixed routing and native CPU controls

Train one 64-centroid CPU Faiss clustering per dataset: seed 20261015, one redo,
12 iterations, disjoint training as above. Freeze FP32 centers, FP64 nearest
assignments and centroid-distance query routing. nprobe = 1/4/16/64, k=10.
Candidate order is centroid rank then ascending source ID within each list.
Every method receives exactly these lists/candidates; verify native index IDs.
Choose nprobe meeting mean recall 0.90/0.95 on calibration queries only, report
evaluation failures without retuning on evaluation data. Retain the whole grid.

Use actual Faiss CPU 1.15.1 (source tag commit
75c755c5a1fc854756aaf3bd1131213aeb21d8d1) implementations, not new replicas:

- IVFFlat candidate-exact reference.
- IVF-PQ, 16 subquantizers, 8 bits/subquantizer; 12 training iterations.
- IVF-RaBitQ 1-bit and 4-bit data, 4-bit query; random rotation seed 123.
  Rotation is applied to codes/queries/centers, while preassigned source IDs and
  probe lists preserve the common routing contract. Check actual list contents.
- For compressed native indexes, retain native top-10 with no raw refinement,
  and exact refinement pools of 10/32/128 returned IDs. No answer-informed
  choice of refinement pool. Report quality/work/storage, NOT CPU/GPU QPS ratios.
  Native search is exhaustive within the selected lists (max_codes=0).

Record code size, serialized index bytes, raw-refinement storage when required,
rotation/codebook storage and CPU build preparation. CPU library results are
algorithm controls, not reproductions of the GPU implementation or its speed.
DARTH and GPU-native implementations are source-audited controls, not executed
baselines in this CPU stage; their performance gates remain explicitly pending.

## Three opportunity levels

1. Perfect answer oracle: only selected-list top-10 payloads read, selector free.
   This is an unattainable ceiling and does NOT identify a realizable mechanism.
2. Final-threshold certificate oracle: evaluate a computable certificate against
   the final selected-list k-th distance. Count evidence and surviving payload
   reads. It knowingly uses hindsight; label it separately from online execution.
3. Online certificate: fully evaluate the first min(64,C) candidates to obtain
   a real threshold. Process remaining candidates in groups of 64. Produce the
   certificate before deciding which payloads to read, refine survivors, then
   update the threshold after that group. No final GT/distance is passed to the
   online selector; oracle distances are used only outside it for validation.

Cheap certificate controls (known methods, not a preselected thesis):

- First D/8 and D/2 raw dimensions: nonnegative partial squared L2 lower bound.
- 8- and 16-group block-norm differences: squared norm-distance lower bound,
  stored FP64 group norms (64/128 additional bytes/vector).
- SQ8 reconstruction plus per-vector FP64 residual norm: triangle lower bound;
  D uint8 codes + 8-byte error norm, shared per-dimension FP64 minimum/scale.
  Train ranges on training data only; clipped outliers retain actual error norms.

Mathematical bounds are deterministic in real arithmetic. Apply a conservative
FP64 numerical guard and verify every evaluated bound against independent exact
distances. Bounded empirical exactness is NOT a universal machine-checked proof.
Reject a known over-tight/invalid-bound mutation and a selector with hidden
final-threshold access. Nonfinite inputs/outputs fail closed.

## Costs, granularity and difficulty

Baseline scan bytes = candidate_count x (4D payload + 8-byte source ID).
Logical read models include evidence, source IDs and admitted payload. For raw
prefixes, report both prefix reuse and full refetch; do not count already-read
dimensions as newly stored evidence. Record mask/selection output bytes separately
and sum them in an explicitly modeled read+write ledger, not GPU physical traffic.
SQ8 still performs D-dimensional reconstruction-distance work; label it, do not
equate fewer input bytes to less arithmetic. Count query evidence preparation
and offline encoding input reads and extra static representation bytes.

Expand surviving candidates to complete 1/8/32-candidate groups as an execution
granularity model. These are NOT measured warp requests or memory transactions.
Retain per-query counts, surviving-ID hashes and 1/8/32-query batch summaries
(mean/max work and repeated-payload union). No GPU p99, cache hit rate or bandwidth
is inferred from these proxies. Stratify queries by held-out global d11/d10 gap
quartiles; these labels are evaluation-only, not an online scheduling oracle.

All representations must disclose storage. An illustrative common static budget
is IVFFlat serialized bytes plus 35% of raw-base payload; flag any inadmissible
variant. Static serialized/storage bytes exclude allocator rounding and runtime
workspaces and are NOT peak VRAM. GPU promotion requires actual peak admission.

## Deliverables and stop rule

Publish only code, input/source manifests, per-query ledgers, an oracle-vs-online
logical-byte plot, a calibrated README and DECISION.md. No unmeasured QPS/p99
plot. Retain failed/incomplete source/logs and model/index files off Git by hash.
Validate small exact examples, negative mutations, native list identity, global
quality, numerical guards, selected-list exactness and an independent CPU check.

At the end of this matrix, record H1/H2/H3/H4 separately as supported, rejected,
partial or unknown. Missing GPU gates do not imply a negative GPU mechanism.
Select ONE next falsification question from the results. Do not turn a perfect
oracle win or standard quantization/filtering into a new CUDA system or paper.
