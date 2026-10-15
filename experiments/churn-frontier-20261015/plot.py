#!/usr/bin/env python3
"""Plot process-level diagnostic observations; error bars are observed ranges."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style_dir = Path.home() / ".agents/skills/paper-figures/scripts"
if style_dir.is_dir():
    sys.path.insert(0, str(style_dir))
    from paper_style import setup_style
    setup_style("acm_double", use_latex=False)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})

NAMES = {"P": "Packed", "F": "Fused (primary)", "I": "Indirect (+64 KiB)"}
COLORS = {"P": "#777777", "F": "#0072B2", "I": "#D55E00"}
MARKERS = {"P": "o", "F": "^", "I": "s"}


def ranges(values):
    median = statistics.median(values)
    return median, [[median-min(values)], [max(values)-median]]


def export(fig, out, name):
    for fmt in ("pdf", "svg", "png"):
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if fmt == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for dim, ax in zip((128, 768), axes):
        recalls = {r["recall"] for r in summary["quality"] if r["dim"] == dim and r["batch"] == 1024 and r["workload"] == "random"}
        constant_quality = len(recalls) == 1
        for variant in "PFI":
            xs, ys, lo, hi = [], [], [], []
            for np in (1, 4, 32):
                rows = [r for r in summary["quality"] if r["dim"] == dim and r["batch"] == 1024 and r["workload"] == "random" and r["variant"] == variant and r["nprobe"] == np]
                assert len(rows) == 4
                assert len({r["recall"] for r in rows}) == 1
                y, error = ranges([32e6/r["query_us"] for r in rows])
                xs.append(np if constant_quality else rows[0]["recall"]); ys.append(y/1000); lo.append(error[0][0]/1000); hi.append(error[1][0]/1000)
            ax.errorbar(xs, ys, yerr=[lo,hi], marker=MARKERS[variant], color=COLORS[variant],
                        label=NAMES[variant], capsize=2, linewidth=1.2, markersize=4)
        layout = next(s for s in summary["shapes"] if s["dim"]==dim)["processes"][0]["variants"]["F"]["layout_bytes"]
        xlabel = f"Partitions scanned (Recall@10={next(iter(recalls)):.3f})" if constant_quality else "Global Recall@10"
        ax.set(title=f"D={dim}; dense layout {layout/2**20:.2f} MiB", xlabel=xlabel, ylabel="Scan + top-k (kquery/s)")
        if constant_quality:
            ax.set_xscale("log",base=2); ax.set_xticks([1,4,32],["1","4","32"])
        else:
            ax.set_xlim(-.02,1.02)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    export(fig, args.out, "quality_memory")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), constrained_layout=True)
    for dim, row in zip((128,768), axes):
        for workload, ax in zip(("fifo","random","hot"), row):
            for variant in "PFI":
                for batch, fill in ((128,"full"),(1024,"none")):
                    shape = next(s for s in summary["shapes"] if (s["dim"],s["batch"],s["workload"])==(dim,batch,workload))
                    values = [p["variants"][variant] for p in shape["processes"]]
                    x, xe = ranges([p["update_us"] for p in values]); y, ye = ranges([32e3/p["query_us"] for p in values])
                    ax.errorbar(x,y,xerr=xe,yerr=ye,marker=MARKERS[variant],color=COLORS[variant],
                                fillstyle=fill,linestyle="none",capsize=2,markersize=4)
            ax.set(title=f"D={dim}, {workload}",xlabel="Relocate + insert (µs)",ylabel="Scan + top-k (kquery/s)")
            ax.margins(x=.15,y=.2)
    fig.legend(handles, labels,loc="outside lower center",ncol=3,frameon=False)
    export(fig,args.out,"update_query_tradeoff")
    (args.out/"CAPTIONS.md").write_text(
        "# Diagnostic figures\n\n"
        "`quality_memory`: random churn, batch 1024, N=8192, 32 queries; points scan 1/4/32 fixed partitions. "
        "Y excludes routing/host work. Each point is the median of four process medians; error bars are their min–max, not a confidence interval. "
        "All variants have the same 16384-row payload budget; indirect adds a 64-KiB row directory. "
        "When recall is constant the x-axis shows the actual probe sweep and states that constant, rather than implying an informative quality frontier. "
        "Layout bytes exclude shared staging, scratch and driver allocations. Synthetic partitions are not trained IVF.\n\n"
        "`update_query_tradeoff`: nprobe=4, same process-level estimator/ranges. Filled markers: batch 128; hollow: batch 1024. "
        "Scatter axes zoom to the observed range; no bar areas or truncated-bar encoding are used. "
        "Only relocate/insert and scan/top-k are timed, not complete dynamic API latency. Equal-quality points are linked to the summary, "
        "not a concurrent service or sustainable churn frontier. The task-owned source plus CPU oracle, not original SIVF, supplies these observations.\n")


if __name__ == "__main__":
    main()
