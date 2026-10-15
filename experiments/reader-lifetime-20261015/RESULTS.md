# Results checkpoint: pressure is known; GPU advantage is unmeasured

## Measured GPU admission, not the pressure matrix

The original cooperative-groups caller completed seven named admission cases:
a two-scope smoke and D=128/768 memcheck, synccheck, and initcheck. Each process
passed protected-payload, acknowledgment and final-capacity assertions. The
unprofiled D128 smoke acknowledged 39 updates while protection was active in
each scope; its logical high-water was 59/58 slots for early/late, with 64
attempts and a 500-us injected protected hold. Both drained to 16 live roots.
These are one-trial smoke observations, not the preregistered 2000-us pressure
matrix, a calibrated performance comparison, or a natural query-tail workload.
Sanitizer timings are never used as latency denominators.

A separate racecheck attempt reported shared-memory hazards in caller/CG and
upstream sites, then failed process isolation. It is **not** a clean race gate
or proof of an upstream algorithmic bug. The caller was changed to a native CTA
barrier/reduction adapter, leaving the upstream header unchanged. Its compile
and smoke completed, but its next memcheck process was marked contaminated;
zero tool errors in an ineligible record do not pass admission. Later live
inspection found another user's compute job on the selected device. GPU work
stopped without touching that job. Exact earlier foreign PIDs were not logged,
so their specific causes remain unresolved. Current-source race localization
and the full 4-process EL/LE matrix remain **unvalidated**, not negative results.

Measured environment: RTX PRO 6000 Blackwell Server Edition, SM 12.0, driver
590.48.01, nvcc 13.1.115, GCC 13.3.0. No clock or power changes. Static compiler
resources are build evidence only, not throughput or safety evidence. Raw
sources, binaries, environments and failure logs remain bound by manifests.

## CPU model: distinguish retention from a leak

The source-matched state-transition model places all 512 writer attempts inside
a held interval. These are **discrete-event model outcomes**, not GPU measurements
or a mapping from microseconds to update counts. Pool slots include 16 current
partition versions. Reader-lifecycle assertions include 100 random legal
8-reader schedules of 2000 updates each; a lost-pinned-version mutation is rejected.

| Scope and held work | Pool | Model first failed attempt (zero-based) | Model acknowledgments / 512 | Model high-water | After drain |
|---|---:|---:|---:|---:|---:|
| Early; pre-read work | 64 | 48 | 48 | 64 | 16 |
| Late; same independent pre-read work | 64 | none | 512 | 19 | 16 |
| Either; actual protected read hold | 64 | 48 | 48 | 64 | 16 |
| Early; pre-read work | 256 | 240 | 240 | 256 | 16 |
| Late; same independent pre-read work | 256 | none | 512 | 19 | 16 |
| Either; actual protected read hold | 256 | 240 | 240 | 256 | 16 |

This predicts a conventional scope effect: delaying acquisition removes only
unnecessary protection, not the lifetime of real reads. Returning to 16 after
release distinguishes grace-period retention from unexplained capacity loss.
In this direct-root model, at most 16 current plus 8 pinned old versions are
semantically needed (plus any in-flight new copy); this is **not** a safe unlink
algorithm or a bound for arbitrary B-tree version-chain traversals.

## What is established versus missing

- Established prior art: GPU scoped snapshots/DEBR and intermediate-version MVGC.
- Partial GPU evidence: native reclaimer API works in the named initial pilot;
  exact source variant and gate boundaries are retained.
- Modeled/inferred: a long protection interval can exhaust a finite pool despite
  no leaked resources; ordinary late acquisition can remove pre-read retention.
- Unknown: current adapter's complete admission, real 2-ms pressure outcomes,
  paired GPU effect, full ANN quality/service p99 and sustainable churn.

No gain over existing GPU scoped-snapshot/EBR has been established. Nor has the
uncollected GPU matrix established parity or failure. Do not convert either
absence into a performance claim.
