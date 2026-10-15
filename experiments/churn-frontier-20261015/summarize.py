#!/usr/bin/env python3
"""Analyze only complete, hash-verified, same-contract diagnostic processes."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required_paths():
    names = {"environment.json", "COMPLETED.json", "build/probe", "build/build.log"}
    names.update("source/" + name for name in ("probe.cu", "run.py", "build.sh", "CONTRACT.md", "PRIOR_ART.md"))
    stems = [f"gate_{tool}_{dim}" for tool in ("memcheck", "synccheck") for dim in (128, 768)]
    stems += ["stress_fifo", "stress_random"]
    stems += [f"p{p}_d{dim}_b{batch}_{workload}" for p in range(4) for dim in (128, 768)
              for batch in (128, 1024) for workload in ("fifo", "random", "hot")]
    names.update(f"raw/{stem}.{suffix}" for stem in stems for suffix in ("csv", "stderr", "start.json", "status.json"))
    return names


def read_run(root):
    completed = json.loads((root / "COMPLETED.json").read_text())
    require(completed["complete"] and completed["mode"] == "full", "full completed run required")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    listed = [row["path"] for row in manifest]
    require(len(listed) == len(set(listed)), "duplicate manifest entries")
    require(required_paths() <= set(listed), "manifest does not cover required evidence")
    for row in manifest:
        name = Path(row["path"])
        require(not name.is_absolute() and ".." not in name.parts, "unsafe manifest path")
        path = root / name
        require(path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"hash mismatch: {name}")
    for field, name in (("source_sha256", "source/probe.cu"), ("binary_sha256", "build/probe")):
        require(completed[field] == hashlib.sha256((root / name).read_bytes()).hexdigest(), "completion digest differs from actual artifact")
    for dim in (128, 768):
        for tool in ("memcheck", "synccheck"):
            stem = f"gate_{tool}_{dim}"
            text = (root / "raw" / f"{stem}.csv").read_text() + (root / "raw" / f"{stem}.stderr").read_text()
            require(re.findall(r"ERROR SUMMARY: (\d+) errors", text) == ["0"], f"missing sanitizer pass: {stem}")
    expected_statuses = 4 + 2 + 4 * 2 * 2 * 3
    statuses = list((root / "raw").glob("*.status.json"))
    require(len(statuses) == expected_statuses, "incomplete process matrix")
    for path in statuses:
        status = json.loads(path.read_text())
        require(status["returncode"] == 0 and not status["contaminated_or_timeout"], f"ineligible process: {path.name}")
        stderr = path.with_name(path.name.replace(".status.json", ".stderr")).read_text()
        require("PASS: live IDs/payloads/ownership" in stderr, "missing oracle outcome")
    return completed


def analyze(root):
    provenance = read_run(root)
    shapes, quality = [], []
    for dim in (128, 768):
        for batch in (128, 1024):
            for workload in ("fifo", "random", "hot"):
                processes = []
                for process, order in enumerate(("PFI", "IFP", "PFI", "IFP")):
                    path = root / "raw" / f"p{process}_d{dim}_b{batch}_{workload}.csv"
                    rows = list(csv.DictReader(path.open()))
                    require(len(rows) == 3 * 80 + 3 * 3 * 20, f"truncated CSV: {path.name}")
                    stats = {}
                    for variant in "PFI":
                        epoch = [r for r in rows if r["kind"] == "epoch" and r["variant"] == variant]
                        require([int(r["step"]) for r in epoch] == list(range(80)), "missing/reordered epochs")
                        require(all(r["order"] == order and int(r["dim"]) == dim and int(r["batch"]) == batch and r["workload"] == workload for r in epoch), "contract drift")
                        require(epoch[-1]["oracle_checked"] == "1", "missing final oracle check")
                        for field in ("update_us", "query_us", "host_plan_us"):
                            require(all(math.isfinite(float(r[field])) and float(r[field]) > 0 for r in epoch), "invalid timing")
                        stats[variant] = {field: statistics.median(float(r[field]) for r in epoch)
                                          for field in ("update_us", "query_us", "host_plan_us", "moved_survivors", "layout_bytes")}
                        stats[variant]["gpu_epoch_us"] = statistics.median(float(r["update_us"]) + float(r["query_us"]) for r in epoch)
                        stats[variant]["gpu_epoch_p99_us"] = sorted(float(r["update_us"]) + float(r["query_us"]) for r in epoch)[math.ceil(0.99*len(epoch))-1]
                        for nprobe in (1, 4, 32):
                            q = [r for r in rows if r["kind"] == "quality" and r["variant"] == variant and int(r["nprobe"]) == nprobe]
                            require(len(q) == 20 and [int(r["step"]) for r in q] == list(range(20)), "missing quality samples")
                            require(len({r["recall"] for r in q}) == 1 and all(r["oracle_checked"] == "1" for r in q), "quality drift")
                            require(all(math.isfinite(float(r["query_us"])) and float(r["query_us"]) > 0 for r in q), "invalid quality timing")
                            require(0 <= float(q[0]["recall"]) <= 1 and (nprobe != 32 or float(q[0]["recall"]) == 1), "invalid full-scan quality")
                            quality.append({"dim": dim, "batch": batch, "workload": workload, "process": process,
                                            "variant": variant, "nprobe": nprobe, "recall": float(q[0]["recall"]),
                                            "query_us": statistics.median(float(r["query_us"]) for r in q),
                                            "layout_bytes": stats[variant]["layout_bytes"]})
                    processes.append({"process": process, "order": order, "variants": stats})
                ratios = {}
                for denominator in ("P", "F"):
                    for metric in ("update_us", "query_us", "gpu_epoch_us"):
                        values = [p["variants"][denominator][metric] / p["variants"]["I"][metric] for p in processes]
                        ratios[f"{denominator}_over_I_{metric}"] = {"geomean": statistics.geometric_mean(values),
                            "process_wins_I": sum(x > 1 for x in values), "min": min(values), "max": max(values),
                            "per_process": values,
                            "order_split": {order: statistics.geometric_mean(v for v,p in zip(values, processes) if p["order"] == order) for order in ("PFI", "IFP")}}
                common_bytes = 28*batch + 4*batch*dim + 4*32*dim + 4*32*32 + 12*32*32*10 + 12*32*10
                shapes.append({"dim": dim, "batch": batch, "workload": workload, "processes": processes,
                               "paired_ratios": ratios, "common_staging_scratch_bytes": common_bytes})
    # All representations must give identical quality for a fixed process/shape/probe.
    keys = {}
    for row in quality:
        key = tuple(row[k] for k in ("dim", "batch", "workload", "process", "nprobe"))
        keys.setdefault(key, set()).add(row["recall"])
    require(all(len(v) == 1 for v in keys.values()), "quality mismatch across representations")
    return {"experiment": "ivf_20261015_payload_mobility_fp32", "scope": "serialized synthetic diagnostic only",
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": provenance["source_sha256"], "binary_sha256": provenance["binary_sha256"],
            "processes_per_shape": 4, "shapes": shapes, "quality": quality,
            "novelty_admitted": False, "sustainable_churn_measured": False,
            "unmeasured_host_costs": ["input generation", "routing", "uploads", "oracle computation"],
            "unvalidated": ["real-data drift", "unbiased temporal routing traces", "trained GPU IVF comparison", "GPU-resident update planning", "concurrent histories", "reader-safe reclamation", "service p99", "Graph mode"]}


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        try:
            analyze(Path(directory))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("empty evidence accepted")
        root = Path(directory)
        completion = {"complete": True, "mode": "full", "source_sha256": "wrong", "binary_sha256": "wrong"}
        (root / "COMPLETED.json").write_text(json.dumps(completion))
        (root / "MANIFEST.json").write_text("[]")
        try:
            read_run(root)
        except ValueError as error:
            assert "cover required evidence" in str(error)
        else:
            raise AssertionError("empty manifest accepted")
        for name in required_paths() - {"COMPLETED.json"}:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
        manifest = [{"path": name, "bytes": (root/name).stat().st_size,
                     "sha256": hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in sorted(required_paths())]
        (root / "MANIFEST.json").write_text(json.dumps(manifest))
        try:
            read_run(root)
        except ValueError as error:
            assert "completion digest" in str(error)
        else:
            raise AssertionError("claimed hashes accepted without artifact binding")
        (root / "MANIFEST.json").write_text(json.dumps(manifest + manifest[:1]))
        try:
            read_run(root)
        except ValueError as error:
            assert "duplicate manifest" in str(error)
        else:
            raise AssertionError("duplicate manifest accepted")
    assert statistics.median([1+9, 9+1, 9+9]) != statistics.median([1,9,9]) + statistics.median([9,1,9])
    print("summary regression checks: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.run:
        result = analyze(args.run.resolve())
        (args.run / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
        for shape in result["shapes"]:
            print(shape["dim"], shape["batch"], shape["workload"], {k: round(v["geomean"], 3) for k,v in shape["paired_ratios"].items()})
