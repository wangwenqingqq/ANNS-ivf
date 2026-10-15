#!/usr/bin/env python3
"""Derive portable summaries from retained raw evidence, without promoting missing gates."""
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from check_state import inspect, inspect_search, require

root = Path(sys.argv[1]).resolve()
required = ["snapshots/smoke/A.json", "snapshots/smoke/search.json",
            "snapshots/T3_growth/growth.csv", "snapshots/T4/overlap.csv",
            "snapshots/T5/publication.csv", "snapshots/T5_injected/publication.csv",
            "snapshots/T7_pool/A.json", "snapshots/T7_pool/B.json", "logs/repro_T7_public.log"]
for case in ("T1_hole", "T1_tail"):
    required += [f"snapshots/{case}/{name}.json" for name in ("A", "B", "B_repeat", "C", "search")]
for case in ("T2_head", "T2_middle"):
    required += [f"snapshots/{case}/{name}.json" for name in ("A", "B", "C", "search_list0", "search_list1")]
for case in ("T3", "T3_counted"):
    required += [f"snapshots/{case}/trials.csv"]
    required += [f"snapshots/{case}/batch_{batch}_trial0.json" for batch in (64, 256, 1024)]
for dim in (128, 256, 257, 960):
    required += [f"logs/memcheck_dim_{dim}.log"]
require(all((root / name).is_file() for name in required),
        "missing required artifacts: " + ", ".join(name for name in required if not (root / name).is_file()))
summary = {"source_commit": (root / "SOURCE_COMMIT.txt").read_text().strip(),
           "evidence_complete": True, "snapshots": {}, "searches": {}, "T3": {}, "T5": {}, "T6": {}}
for path in sorted((root / "snapshots").rglob("*.json")):
    state = json.loads(path.read_text())
    key = str(path.relative_to(root))
    if "slabs" in state:
        summary["snapshots"][key] = inspect(state)
    elif "reference" in state:
        summary["searches"][key] = inspect_search(state)
for path in sorted((root / "snapshots").glob("T3*/trials.csv")):
    grouped = defaultdict(list)
    for row in csv.DictReader(path.open()):
        grouped[int(row["batch"])].append(row)
    require(set(grouped) == {64, 256, 1024}, f"{path.name}: missing T3 batch")
    summary["T3"][path.parent.name] = {}
    for batch, rows in grouped.items():
        require([int(r["trial"]) for r in rows] == list(range(1000)), "incomplete/reordered T3 trials")
        data = {"trials": len(rows), "trials_with_orphans": sum(int(r["orphan"]) > 0 for r in rows),
                "orphan_min": min(int(r["orphan"]) for r in rows),
                "orphan_max": max(int(r["orphan"]) for r in rows),
                "orphan_total": sum(int(r["orphan"]) for r in rows),
                "free_min": min(int(r["free"]) for r in rows),
                "committed_min": min(int(r["committed"]) for r in rows)}
        if rows[0].get("head_cas_failed", "NA") != "NA":
            data["orphan_equals_failed_cas_every_trial"] = all(int(r["orphan"]) == int(r["head_cas_failed"]) for r in rows)
            data["allocations_equal_reachable_plus_orphans_every_trial"] = all(int(r["slab_allocated"]) == int(r["orphan"]) + int(r["reachable"]) for r in rows)
            data["reclaimed_total"] = sum(int(r["slab_reclaimed"]) for r in rows)
            data["insert_failed_total"] = sum(int(r["insert_failed"]) for r in rows)
        summary["T3"][path.parent.name][batch] = data
growth = root / "snapshots/T3_growth/growth.csv"
if growth.exists():
    rows = list(csv.DictReader(growth.open()))
    require([int(r["batch"]) for r in rows] == list(range(1000)), "incomplete growth batches")
    summary["T3_growth"] = {"batches": len(rows), "first": rows[0], "last": rows[-1],
                            "orphans_nondecreasing": all(int(a["orphan"]) <= int(b["orphan"]) for a, b in zip(rows, rows[1:]))}
overlap = root / "snapshots/T4/overlap.csv"
if overlap.exists():
    rows = list(csv.DictReader(overlap.open()))
    require([int(r["trial"]) for r in rows] == list(range(1000)), "incomplete overlap trials")
    summary["T4"] = {"trials": len(rows), "inconsistent_returned_pairs": sum(int(r["payload_mismatch"]) for r in rows),
                     "trials_with_three_overlapping_event_windows": sum(max(float(r[f"{op}_start_ms"]) for op in ("search", "insert", "delete")) < min(float(r[f"{op}_end_ms"]) for op in ("search", "insert", "delete")) for r in rows),
                     "scope": "independent streams with a common gate; event windows are bounds, not an instruction-level overlap trace"}
for path in (root / "snapshots").glob("T5*/publication.csv"):
    rows = list(csv.DictReader(path.open()))
    expected_trials = 10 if path.parent.name == "T5_injected" else 1000
    require([int(r["trial"]) for r in rows] == list(range(expected_trials)), "incomplete publication trials")
    summary["T5"][path.parent.name] = {"trials": len(rows), "stale_allocations": sum(int(r["allocated"]) == 0 for r in rows),
                                      "correct_allocations": sum(int(r["allocated"]) == 1 for r in rows),
                                      "empty_returns": sum(int(r["allocated"]) == -1 for r in rows)}
for dim in (128, 256, 257, 960):
    log = root / f"logs/memcheck_dim_{dim}.log"
    if log.exists():
        text = log.read_text()
        errors = re.findall(r"ERROR SUMMARY: (\d+) errors", text)
        require(bool(errors) or "HARNESS_ERROR" in text or "ERROR" in text, "incomplete sanitizer log")
        if "exact_topk=PASS" in text:
            require((root / f"snapshots/dim_{dim}/search.json").is_file(), "missing dimension search result")
        summary["T6"][dim] = {"memcheck_errors": int(errors[-1]) if errors else None,
                               "search_passed": "exact_topk=PASS" in text,
                               "scope": "source bounds risk is not itself a reproduced device memory error"}
for case in ("T1_hole", "T1_tail"):
    folder = root / "snapshots" / case
    if (folder / "B_repeat.json").exists():
        summary.setdefault("I6", {})[case] = json.loads((folder / "B.json").read_text()) == json.loads((folder / "B_repeat.json").read_text())
public = root / "logs/repro_T7_public.log"
if public.exists():
    public_text = public.read_text()
    summary["T7_public"] = [line for line in public_text.splitlines() if line.startswith("T7_PUBLIC")]
    require(bool(summary["T7_public"]) or public_text.startswith("Unresolved:"), "missing public API outcome")
controls = ["snapshots/smoke/A.json"]
for case in ("T1_hole", "T1_tail"):
    controls += [f"snapshots/{case}/{name}.json" for name in ("A", "B", "B_repeat")]
controls += ["snapshots/T1_tail/C.json", "snapshots/T2_head/A.json", "snapshots/T2_middle/A.json"]
controls_ok = all(summary["snapshots"][key]["passed"] for key in controls)
controls_ok &= summary["searches"]["snapshots/smoke/search.json"]["passed"]
controls_ok &= summary["searches"]["snapshots/T1_tail/search.json"]["passed"]
counterexamples = {
    "T1": not summary["snapshots"]["snapshots/T1_hole/C.json"]["passed"],
    "T2_head": not summary["snapshots"]["snapshots/T2_head/B.json"]["passed"],
    "T2_middle": not summary["snapshots"]["snapshots/T2_middle/B.json"]["passed"],
    "T3": any(row["trials_with_orphans"] > 0 for row in summary["T3"]["T3"].values()),
}
summary["assessment"] = {"controls_passed": controls_ok, "counterexamples_reproduced": counterexamples,
                         "status": "inconclusive_controls_failed" if not controls_ok else
                                   "counterexamples_reproduced" if any(counterexamples.values()) else
                                   "no_counterexample_in_tested_cases",
                         "index_validated": False, "throughput_gate": "closed"}
(root / "EVIDENCE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
manifest = []
for directory in ("logs", "snapshots", "build", "harness"):
    for path in sorted((root / directory).rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(root / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key not in ("snapshots", "searches")}, indent=2))
