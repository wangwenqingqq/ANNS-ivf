# Reader-lifetime diagnostic: preregistration

Date: 2026-10-15. Parent ANNS-ivf commit: 59f6ce7abf5d23f1290b02d6fee780117667d2f7.
Experiment: ivf_20261015_reader_lifetime_native_debr. State: designed.

## Question and novelty gate

Does an overlapping reader's protection interval create measurable version-pool
pressure, and does ordinary just-in-time GPU operation scoping remove it?
This is a component-level kill test, not a new reclamation protocol or ANN index.
The exact upstream DEBR_device implementation is the comparator, not a rewritten
EBR approximation. Its upstream commit is f755006d78ccca565f97cda053727d4d48b82d46
(owensgroup/MVGpuBTree). Headers remain unmodified and are fetched reproducibly.
GPU synchronization-scoped snapshots are established (PACT 2022), and reclaiming
unneeded intermediate versions is established (DISC 2021). Neither is admitted
as a novel thesis. No paper drafting or expensive ANN scaling in this round.

## Observable semantics and fairness

A reader obtains one immutable partition version by an acquire root load while
inside the native EBR critical section. Its entire read observes that generation
until release. A writer initializes a replacement payload, release-publishes its
root, then retires the old address through the upstream reclaimer. Publication
is the successful update's linearization/acknowledgment point. A failed allocation
is explicitly reported, not counted as an effective update. Queries may legally
linearize at any root load between invocation and response. No atomic multi-list
snapshot, transactional update batch, full MVGpuBTree traversal, or complete ANN
search/Recall@10 claim is made.

Two ordinary scopes share the same payload/allocator/reclaimer/read work:
- early: enter EBR and acquire the version before independent pre-read work;
- late: perform the same independent work, then enter EBR and acquire the version.
They can return different legal generations. Requiring invocation-time snapshots
would disallow the late control; it is explicitly not the contract here.
Both hold protection through their final actual payload read. No subdivision of
a live query, unsafe release/reacquire, or weaker read semantics is permitted.
An in-critical-section hold has identical placement for both controls.

## Workload and resource scope

One cooperative persistent CUDA grid: eight reader CTAs and one writer CTA,
128 threads each. This guarantees residency, not arbitrary oversubscribed-kernel
progress. Sixteen current partition roots; each immutable partition contains
64 FP32 vectors, D=128 or 768. Reader CTA i reads root i, including untouched
payload-word/generation integrity checks. The writer replaces partition roots
round-robin for 512 allocation attempts. This is an intentionally synthetic COW
partition model; bytes per replacement are not asserted unavoidable for ANN.
Fixed pools: 64 or 256 total version slots (16 initially live). Compare equal
bytes within each shape. Shared/overflow DEBR bags are sized identically with
an explicit capacity bound; no silent bag overflow. Metadata, payload allocation,
logical high-water and reclaim/drain counts are separate quantities.

Scenarios: no injected wait; 2000-us independent pre-read wait; 2000-us wait
inside the protected read. These waits are controlled sensitivity injections,
NOT a measured real query-tail distribution. Four payload scan repetitions.
The zero-wait control provides the unforced case. Reader and writer handshakes
establish overlap opportunities; actual acknowledged updates during protection
are recorded, not assumed from concurrent launch. Initialization and final drain
are synchronized; update/read operations between them genuinely overlap.

## Measurement and gates

GPU 6 only if live checks show no other compute process; same per-device advisory
lock as the previous campaign; abort our own child on foreign compute/timeout.
Do not alter power, clocks, other processes, or the canonical checkout.
SM 12.0, CUDA C++20, -O3 -arch=sm_120 -lineinfo --expt-relaxed-constexpr.
Record actual driver, compiler, power and clocks. No throughput promotion.

Before collection: build; small native EBR smoke; memcheck, synccheck and initcheck
for both ordinary scopes; internal assertions detect early free, double free,
capacity loss, generation/payload mismatch, and missing final roots. Inspect
static resources. Then four fresh processes with EL/LE/EL/LE order. Each shape
has one warmup and four retained independent reset trials per scope. Preserve
all failure records, per-attempt observations, reader intervals and order.
Primary results: first capacity failure, acknowledged versus failed updates,
version high-water, and post-reader drain to exactly 16 live roots. Timing is
supporting diagnostic wall/device-span information, not an ANN speedup. Report
process medians/min-max and all individual outcomes, not confidence of a new win.

If the native comparator fails any invariant, stop timing and diagnose the exact
adapter/upstream boundary; do not silently replace its code or call it unsafe.
No clock injection result may be promoted to sustainable churn or production p99.

## Design card and stop/reopen condition

The writer CTA owns allocation search, publication and retirement. Each reader
CTA owns a single acquired root and a fixed payload read; the upstream block-wide
reclaimer owns bag rotation/epoch advancement. Global root/control words use
CUDA device-scope acquire/release atomics. Cooperative grid barriers delimit
initialization/final drain only; CTA barriers delimit read completion before
reclaimer release. A reader ref counter is a diagnostic early-free assertion,
not the reclamation decision. Phase live state is one acquired slot/generation,
scalar checksum and word-validation flag; no tensor or shared payload staging.

Predicted delta: moving only pre-read work out of the critical section can reduce
unnecessary epoch retention without changing bytes copied/read. A long actual
protected read remains a barrier for global EBR. If this is all that is observed,
reject lifetime/scoping as a new thesis and switch questions. A positive remaining
gap still must beat known MVGC/hazard/precise-retention techniques and survive
real-data/ANN same-contract evidence; this microtest cannot admit that mechanism.

## Admission refinement (before full collection)

The first pilot used CUDA cooperative_groups thread_block_tile<128>. Its
memcheck/synccheck/initcheck and oracle passed. A separate racecheck attempt
reported both caller and upstream/CG shared hazards, then failed the strict
process-isolation monitor; it is ineligible evidence, not a clean race gate.
Before any full collection, replace only the caller's tile adapter with a
128-thread BlockGroup using native __syncthreads / __syncthreads_and. These are
the same CTA-wide synchronization/reduction semantics required by the templated
upstream API. The DEBR header remains byte-identical. Re-run admission and
racecheck to localize the boundary; do not suppress warnings or call a shared
upstream warning a new-protocol advantage. The original pilot is preserved.
The runner also checks child completion after its polling sleep, before polling
GPU PIDs, to avoid classifying a completed child's stale driver entry as foreign;
unknown ancestry while the child remains active still fails closed.

## Resource checkpoint and added race gate

Before a full matrix, include racecheck on both dimensions (8 attempts per scope
to keep instrumentation bounded) and require zero reported hazards. This is an
additional gate, not a retroactive pass for the first pilot. The native-CTA
adapter's race gate is pending. If only common upstream same-value shared writes
remain, retain and audit them; no waiver or baseline promotion is automatic.

During admission, another user's vLLM occupied the selected GPU. Stop GPU work,
leave that process untouched, and request an available device. The two rejected
monitoring records lack an exact foreign PID, so do not infer their exact cause
from the later live observation. They remain ineligible. The runner now retains
foreign PID/status diagnostics privately on future isolation failures. This is
not permission to pollute public evidence with other users' process metadata.
