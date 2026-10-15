# Churn layout diagnostic: frozen contract

Experiment: `ivf_20261015_payload_mobility_fp32`. State: designed, diagnostic-only.
Parent source: `d1ec8cfd7a7ee609b0fd0c63ce6c18fb04975654`. Original SIVF is
neither repaired nor used as a valid performance comparator.

## Question, alternatives, and stop rule

Does avoiding relocation of surviving payloads pay for indirect query reads,
**after** a competent dense baseline fuses delete/insert into the same epoch?

- `packed`: dense per-list FP32 rows; batch deletion compacts, then appends.
- `fused`: same dense representation, but incoming rows first fill holes and
  only unavoidable out-of-range survivors relocate. This is the primary control.
- `indirect`: a compact per-list row directory points to a stable global payload
  pool; survivor relocation moves pointers, not vectors. This is a diagnostic
  representation, **not an admitted novel research thesis**.
- Strong competing explanation: apparent savings against `packed` disappear
  against ordinary fused hole reuse. Reject the broad mobility thesis if that
  happens; do not relabel the weaker comparator as the main baseline.
- Earlier SIVF correctness failures do not veto timings of these separate,
  oracle-checked representations. They still veto original-SIVF throughput claims.

## Fixed workload

N=8192 live vectors; 32 fixed routing partitions; capacity 512 rows/partition;
physical payload capacity 16384 rows for **all** variants. FP32, row-major, exact
L2 within the same probed partition set; k=10; query batch=32. Dimensions 128 and
768; churn batches 128 and 1024; workloads FIFO, random, hotspot. Synthetic,
deterministic hash-derived inputs and explicitly assigned partitions, not trained
IVF, a public dataset, or an external-library comparison.

Deletion IDs are unique and selected from the current live oracle. FIFO removes
the oldest IDs; random samples without replacement; hotspot biases the first
four partitions. New vectors route to the next partition cyclically (within the
four-hot or 28-cold group for hotspot). This changes per-list counts and avoids
the unfair same-list replacement case where dense hole reuse needs no movement.
Overflow of any declared capacity or a missing/duplicate ID aborts the process.

Each epoch is a **serialized quiescent transaction**: delete+insert commits before
the following queries. No concurrent query, linearizability, reader lifetime,
or sustainable service-throughput claim is supported. The macro-epoch fusion
contract applies equally to all variants. Live count is constant; unique IDs,
payloads, ownership and directory coordinates must match the CPU oracle.

## Measurement and evidence gates

- One SM 12.0 GPU, compiler CUDA 13.1, `-O3 -std=c++20 -arch=sm_120 -lineinfo`;
  record actual device/driver/power/clocks, not assumed fixed clocks.
- Select GPU 6 only after checking all compute processes and lock holders. Hold
  one campaign lock. Never stop processes or change host/power configuration.
- Correctness: validate every live record/payload and query at beginning/end;
  compare exact selected-list top-10 with CPU and report global Recall@10.
- Admission: standalone small correctness test, memcheck and synccheck on both
  dimensions; no result promotion until these pass. Then bounded stress checks.
- Four fresh processes; alternating variant order `PFI`, `IFP`, `PFI`, `IFP`;
  seed fixed per contract. Warmup 20, retain 80 epochs per shape. All raw samples
  and order are retained. Per-process medians are the analysis unit. Report the
  geometric mean of paired process ratios and process wins; do not imply a
  high-confidence production win from four diagnostic processes.
- Primary scope: CUDA-event update (survivor relocation + insertion) and scan
  plus top-k; their sum is a **GPU serialized epoch**, not end-to-end latency.
  Host planning, routing, input generation/uploads and oracle work are outside
  these timers. Only host planning is timed separately; the other host costs
  remain explicitly unmeasured. No achieved sustainable churn number.
- Query quality curve: nprobe 1/4/32 on the final oracle, same queries/partitions
  and FP32 semantics across representations. No reduced-recall speedup.
- Memory: explicitly allocated layout bytes, common staging/scratch bytes and
  fixed capacity; not process-wide peak VRAM attributed to one variant when all
  three coexist. Resource sustainability is only bounded pool reuse in this
  serialized model, not an online reclamation result.
- Stop on any correctness/sanitizer/capacity failure or another compute process
  on the selected device. Preserve failed artifacts and do not publish their
  timings as valid measurements. No large dataset, multi-GPU or paper drafting.

## Kernel design card

| Kernel | Ownership / layout | Readiness / synchronization | Mechanism |
|---|---|---|---|
| relocate | One CTA per moved vector, contiguous dimensions; indirect moves one row pointer | Disjoint destinations; donor sources outside compacted prefix; completes before insert kernel | Full-payload vs metadata-only movement |
| insert | One CTA per new vector; resident input payload; direct row or global row + directory | Disjoint destinations and physical pool rows; stream order after relocation | Same payload work across variants |
| partial top-k | One CTA per query/probed partition; 8 warps, one vector per warp | Warp shuffle L2 sum; 8 private top-10 arrays then shared merge, CTA barriers | Same scan algorithm, only row addressing changes |
| merge top-k | One CTA per query; thread 0 merges partition top-10 | Same stream after all partial kernels | Identical final result contract |

No Tensor Cores, TMA, TMEM, shared pipeline or persistent rollover is introduced.
Registers hold per-warp top-10 state; shared state is 8 top-10 lists. Payload is
read before a warp advances; shared outputs are overwritten only after the CTA
barrier and consumed before the kernel exits. This is not a cuVS-competitive
search implementation. The lower-bound hypothesis is extra moved bytes versus
directory loads/random-row locality, not novel instruction throughput.

## Claim states before execution

| Claim | State | Evidence | Allowed wording |
|---|---|---|---|
| COW slab publication is a new direction | rejected | ETALE | Not novel by itself |
| Batch ownership/merge is a new direction | rejected | GrAND, GPMA+ | Not novel by itself |
| Indirection alone is a new systems contribution | rejected | Established row-directory/gather design | Diagnostic control only |
| Survivor movement dominates competent dynamic IVF | unknown | No new measurement yet | Hypothesis under test |
| Stable payload beats fused packing at matched quality | unknown | No new measurement yet | Hypothesis under test |
| Online C/Q/U/M/X/N targets achieved | unknown | Not tested | Do not claim |

## Predeclared implementation refinement after the pilot

The initial indirect mover assigns a CTA to each pointer (matching the vector
mover's launch geometry). This is avoidable overhead, not an inherent cost of
indirection. Preserve `pilot_01` and `full_01` as the original implementation;
`full_02` uses one thread per directory move, with all other contracts unchanged.
The source and binary hashes distinguish the two. Compare against freshly
remeasured `packed`/`fused` in the same second campaign. Do not promote a negative
result from the inefficient mover into a mechanism-level rejection, or promote
this ordinary launch fix into novelty. Both full runs retain four process orders.

Accounting clarification before analysis: the executable reports layout bytes
and host-plan time. Common GPU staging/scratch is derived from explicit buffer
allocations as `28*B + 4*B*D + 4*Q*D + 4*Q*L + 12*Q*L*K + 12*Q*K` bytes. Driver,
event, host-oracle and unmeasured host costs are not represented by those bytes.
This corrects an overly broad reporting promise in the initial record; it does
not change timed code or retrofit an end-to-end measurement.

Attribution clarification: `packed` and `indirect` preserve the same logical
scan order, so their scan ratio is the cleaner addressing/locality contrast.
`fused` can permute surviving rows differently while preserving the exact same
candidate set and results. Its total-epoch comparison is the competent-control
comparison, not a claim that every query-time difference is caused by one
directory load. Optional CUPTI tracing diagnoses kernel/launch costs only; its
durations never replace the CUDA-event denominator.

Post-run controller hardening: a reported GPU PID whose ancestry cannot be read
is now classified as foreign (fail closed), with a CPU regression test. The
recorded campaigns retain their original controller bytes; their periodic
visible-process checks are not retroactively upgraded into a strict allocation
guarantee. Timed kernel source and measurements are unchanged.
