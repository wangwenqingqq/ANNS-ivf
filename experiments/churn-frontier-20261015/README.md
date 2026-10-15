# Open exploration: payload mobility versus query locality

This round asks a mechanism question, **not** how to finish repairing SIVF:

> When deletions and insertions continuously change IVF membership, is avoiding
> survivor-payload relocation worth the resulting query indirection after a
> competent dense implementation already fuses hole reuse?

Read `PRIOR_ART.md` first. COW slabs, GPU epochs, batch ownership, adaptive
maintenance, and indirection alone are **not admitted as new contributions**.
The three oracle-checked layouts are diagnostic controls, not a production index.
`fused` is the primary performance control; `packed` is an ablation that can
overstate the benefit of avoiding movement.

## Reproduce

From the repository root, on an unused SM 12.0 GPU with CUDA 13.1, Python 3,
Compute Sanitizer and `nvidia-smi`:

```bash
EXP=experiments/churn-frontier-20261015
python3 "$EXP/run.py" --self-test
python3 "$EXP/summarize.py" --self-test
python3 "$EXP/run.py" --gpu 6
python3 "$EXP/summarize.py" "$EXP/runs/<new-run-directory>"
```

The default run compiles a frozen copy of the source, checks memcheck/synccheck
at D=128/768, performs two 1000-epoch bounded-pool stresses, then collects four
fresh, order-balanced processes per shape. Existing output directories are never
overwritten. A busy GPU, nonzero process exit, timeout or foreign compute process
closes the run. Only a process launched by this runner may be terminated.

For a short admission pilot, add `--pilot`. Its results cannot pass the full
summary gate or support an all-shape conclusion. Single manual invocations of
`build/probe` do not acquire a device lock; use the runner for actual collection.

Plotting is optional and does not affect experiment correctness:

```bash
python3 -m pip install -r "$EXP/requirements-plot.txt"
python3 "$EXP/plot.py" "$EXP/runs/<new-run-directory>/SUMMARY.json" /tmp/churn-plots
```

## Scope that must not be expanded

- Synthetic explicitly assigned routing partitions, not trained IVF or a public
  dataset. Exact selected-partition top-k is checked against a CPU oracle;
  reported global recall is measured for the synthetic oracle only.
  Initial IDs are list-major, so the FIFO trace has age/partition correlation;
  it is an explanatory case, not a representative real streaming trace. Random
  and hotspot cases do not eliminate the need for real temporal data later.
- Serialized macro-epochs, not overlapping queries/updates. No sustainable churn
  throughput, concurrent snapshot guarantee, online-reclamation proof or service
  p99 is measured.
- Timers cover relocation+insertion and scan+top-k. Host plans are timed outside
  them. Input generation, routing, uploads and CPU oracle costs are unmeasured.
  In particular, the current host planner is not a GPU-resident update path.
- All three layouts coexist in a process. Per-layout allocation bytes and common
  workspace are accounted explicitly, not confused with process VRAM attribution.
- Correctness assertions cover initialization and selected synchronized endpoints;
  they do not rule out arbitrary intermediate or untested concurrent failures.
- Four process medians/ranges are diagnostic evidence, not a production speedup
  confidence claim or attainment of the brief's 90%/2× exploration targets.

## Artifacts and versions

`CONTRACT.md` freezes shapes, semantics, estimator, stop/reopen boundaries and the
standard directory-mover refinement. Raw runs, binaries, commands and environment
records remain under ignored `runs/`; `MANIFEST.json` binds all raw bytes.
The summary rejects incomplete matrices, unbound source/binary hashes, duplicate
manifest paths, failed gates and unmatched quality.

Only curated data, code, figures and decision notes belong in Git. Raw command
metadata embeds execution-checkout paths and must not be force-added. The old
SIVF audit and its original source remain intact on their own branch.
