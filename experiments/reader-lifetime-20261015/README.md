# Reader lifetime: native GPU EBR kill test

**Checkpoint, not a completed GPU comparison.** The standalone early/late-scope
experiment calls the unmodified upstream GPU DEBR reclaimer at a pinned version.
CPU transition checks pass. The initial native-CG admission passed its named
oracle/memcheck/synccheck/initcheck gates; racecheck and the current CTA adapter's
admission are not complete. A foreign job occupied the selected device during
admission, so the 4-process pressure matrix was not collected.

Read `CONTRACT.md`, `PRIOR_ART.md`, `RESULTS.md`, and `DECISION.md`. The original
SIVF and previous payload-mobility experiment are not modified. No new mechanism,
ANN speedup, sustainable churn result, or proof of native-library unsafety is claimed.

## Reproduce CPU checks

From the repository root (Python standard library only):

```bash
EXP=experiments/reader-lifetime-20261015
python3 "$EXP/run.py" --self-test
python3 "$EXP/model.py" --self-test --out /tmp/reader-lifetime-model.json
python3 "$EXP/summarize.py" --self-test
```

The model's event-count schedules are not measured GPU timings. Its all-held
schedule is deliberately adversarial; it does not turn 2000 us into a fixed
number of updates. Random legal schedules and a premature-free negative control
exercise its accounting assertions. It does not model CUDA memory ordering.

## Resume GPU admission before measurement

Require an unused SM 12.0 GPU, CUDA 13.1, Compute Sanitizer, Git and Python 3.
`build.sh` fetches the pinned public headers into ignored `external/`; it does
not install packages or alter the upstream source. A local Git bundle of the
same commit may seed `external/MVGpuBTree` when remote HTTPS is unavailable.
The original Apache-2.0 copyright/license remains in that fetched repository;
no third-party source or binary is republished here.

```bash
bash "$EXP/build.sh"                          # compile only, no device execution
python3 "$EXP/run.py" --gpu 6 --pilot         # only after this GPU is available
python3 "$EXP/run.py" --gpu 6                 # admission + full matrix, fail closed
python3 "$EXP/summarize.py" "$EXP/runs/<completed-full-run>" > /tmp/reader-summary.json
```

No existing output directory is overwritten. The runner shares the previous
experiment's per-device lock and ancestry-isolation helper. Unknown foreign
processes/timeouts stop only the launched child. It never changes GPU power or
clocks or terminates another user's job. Raw command/status records may include
private paths/process metadata: keep `runs/` ignored. A successful compiler or
partial sanitizer log is not permission to bypass a failed isolation/race gate.

The full analyzer rejects pilots, missing matrices, altered artifacts, upstream
mismatches and failed gates. A legacy `--admission` read deliberately reports only
the initial seven named gates and never means racecheck or full admission passed.

## Artifact states

- `MODEL_RESULTS.json`: reproducible CPU model only.
- `ADMISSION_INITIAL.json`: digest-bound initial native-CG pilot, not current source.
- `variants/initial_group.patch`: current source to that initial source exactly.
- `evidence/`: curated numerical pilot records and raw-file hash manifests;
  rejected isolation/race records remain labeled ineligible, not silently dropped.
- Current `probe.cu`: native CTA group adapter; built/smoke-observed, final
  isolation/sanitizer admission pending. `UPSTREAM.json` binds unmodified headers.
