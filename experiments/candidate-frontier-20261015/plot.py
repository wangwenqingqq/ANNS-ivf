#!/usr/bin/env python3
"""Plot frozen logical-byte receipts, never unmeasured GPU performance."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root):
    evidence = root / "evidence"
    results = json.loads((evidence / "RESULTS.json").read_text())
    with gzip.open(evidence / "QUERY.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    skills = Path(os.environ.get("AGENTS_SKILLS_HOME", Path.home() / ".agents" / "skills"))
    sys.path.insert(0, str(skills / "paper-figures" / "scripts"))
    try:
        from paper_style import setup_style
        setup_style("acm_double", use_latex=False)
    except ImportError:
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.85), sharey=True)
    names = ("prefix_8", "prefix_2", "norm_8", "norm_16", "sq8")
    labels = ("D/8\nprefix", "D/2\nprefix", "8 block\nnorms", "16 block\nnorms", "SQ8 +\nerror")
    ledger, diagnostics = [], {}
    for ax, ds in zip(axes, results["datasets"]):
        name = ds["name"]; p = ds["calibration"]["0.95"]["chosen_nprobe"]
        subset = [r for r in rows if r["dataset"] == name and r["nprobe"] == p]
        selected = lambda m: [r for r in subset if r["method"] == m]
        average = lambda values: float(np.mean(values))
        perfect = average([100 * (1 - r["read_bytes"] / r["baseline_candidate_read_bytes"]) for r in selected("perfect_answer_oracle")])
        ax.axhline(perfect, color="black", linestyle="--", linewidth=1, label="Perfect answer oracle")
        settings = (("final_tau_oracle", None, "#777777", "o", "Final-threshold oracle"),
                    ("online", None, "#0077BB", "s", "Online, candidate grain 1"),
                    ("online", "32", "#EE7733", "^", "Online, candidate grain 32"))
        for mode, grain, color, marker, label in settings:
            values = []
            for m in names:
                rs = selected(m + ":" + mode)
                values.append(average([100 * (1 - (r["grain_models"][grain]["read_bytes_refetch"] if grain else r["read_bytes_refetch"]) / r["baseline_candidate_read_bytes"]) for r in rs]))
            ax.plot(range(5), values, marker=marker, color=color, linewidth=1.2, markersize=4,
                    linestyle=":" if grain else "-", label=label)
        ax.axhline(0, color="#BBBBBB", linewidth=.6)
        ax.set_xticks(range(5), labels, fontsize=7)
        ax.set_ylim(-58, 105); ax.set_yticks([-50, 0, 50, 100]); ax.set_xlim(-.3, 4.3)
        ax.set_title(f"{name}{' (synthetic)' if ds['synthetic'] else ''}\nD={ds['d']}, nprobe={p}, R@10={ds['calibration']['0.95']['evaluation_recall']:.3f}", fontsize=8)
        diagnostics[name] = {"chosen_nprobe": p, "query_count": 64, "perfect_oracle_avoided_percent": perfect, "certificates": {}}
        for m in names:
            rs = selected(m + ":online")
            dsdiag = {"online_read_ratio": average([r["read_bytes_refetch"] / r["baseline_candidate_read_bytes"] for r in rs]),
                      "grain32_read_ratio": average([r["grain_models"]["32"]["read_bytes_refetch"] / r["baseline_candidate_read_bytes"] for r in rs]),
                      "full_row_fraction_quartiles": np.quantile([r["full_rows"] / r["candidates"] for r in rs], [.25, .5, .75]).tolist(),
                      "hardness_quartiles": {str(q): {"mean_full_rows": average([r["full_rows"] for r in rs if r["gap_quartile_0_hardest"] == q]), "mean_read_ratio": average([r["read_bytes_refetch"] / r["baseline_candidate_read_bytes"] for r in rs if r["gap_quartile_0_hardest"] == q])} for q in range(4)}}
            diagnostics[name]["certificates"][m] = dsdiag
        keys = sorted({(r["method"], r.get("refine_pool", -1)) for r in subset})
        for m, refine in keys:
            rs = [r for r in subset if r["method"] == m and r.get("refine_pool", -1) == refine]
            footprint = ds["storage"][m.split(":")[0]].get("declared_static_storage_bytes") if ":" in m else None
            ledger.append({"dataset": name, "nprobe": p, "method": m, "refine_pool": refine,
                "mean_candidates": average([r["candidates"] for r in rs]),
                "global_recall_at_10": average([r["global_recall"] for r in rs]),
                "mean_full_payload_rows": average([r.get("full_rows", 10) for r in rs]),
                "mean_evidence_or_code_read_bytes": average([(0 if m == "flat" else r.get("evidence_read_bytes", r.get("code_read_bytes_model", 0))) for r in rs]),
                "mean_payload_read_bytes": average([(r["candidates"] * ds["d"] * 4 if m == "flat" else r.get("raw_payload_read_bytes", r.get("raw_refinement_read_bytes", 10 * ds["d"] * 4))) for r in rs]),
                "mean_source_id_read_bytes": average([r["candidates"] * 8 for r in rs]),
                "mean_logical_total_read_bytes": average([r.get("read_bytes_refetch", r.get("read_bytes")) for r in rs]),
                "mean_query_read_ratio_to_flat": average([r.get("read_bytes_refetch", r.get("read_bytes")) / r["baseline_candidate_read_bytes"] for r in rs]),
                "mean_mask_queue_write_bytes_model": average([r.get("modeled_mask_write_bytes", 0) + r.get("modeled_queue_write_bytes", 0) for r in rs]) if ":" in m else "not modeled",
                "declared_static_bytes": footprint if footprint else rs[0].get("static_storage_bytes", "oracle: not realizable"),
                "kernel_time": "unmeasured", "gpu_physical_bytes": "unmeasured", "peak_vram": "unmeasured"})
    axes[0].set_ylabel("Modeled candidate bytes avoided (%)", fontsize=8)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", bbox_to_anchor=(.52, 1.02), ncol=2, frameon=False, fontsize=7)
    fig.subplots_adjust(left=.09, right=.99, top=.72, bottom=.23, wspace=.12)
    fig.text(.5, .035, "64 held-out queries; mean per-query ratio. Read model only, not GPU traffic or speed.", ha="center", fontsize=7)
    folder = root / "figures"; folder.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = folder / ("opportunity." + suffix)
        fig.savefig(path, dpi=300, metadata={"Creator": "Matplotlib"})
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    with (root / "WORK.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=ledger[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(ledger)
    (evidence / "DIAGNOSTICS.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().root)
