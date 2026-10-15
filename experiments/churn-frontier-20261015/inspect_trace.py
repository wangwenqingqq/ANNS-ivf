#!/usr/bin/env python3
"""Summarize a separate CUPTI trace; never use it as the latency denominator."""
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys

path, output = map(Path, sys.argv[1:])
connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
rows = connection.execute("""SELECT k.start,k.end,k.gridX,k.streamId,s.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
    ORDER BY k.start""").fetchall()
groups = {}
for index, (start, end, grid, stream, name) in enumerate(rows):
    match = re.search(r"scan<\(bool\)([01])>", name)
    if not match:
        continue
    assert grid % 32 == 0 and end > start
    following = rows[index+1]
    assert "merge(" in following[4] and following[3] == stream and following[0] >= end
    family = "indirect" if match[1] == "1" else "direct_P_and_F"
    key = f"{family}_nprobe{grid//32}"
    groups.setdefault(key, []).append({"scan_us": (end-start)/1000,
        "merge_us": (following[1]-following[0])/1000,
        "interkernel_gap_us": (following[0]-end)/1000,
        "same_trace_device_span_us": (following[1]-start)/1000})
assert set(groups) == {f"{family}_nprobe{np}" for family in ("indirect","direct_P_and_F") for np in (1,4,32)}
summary = {"scope": "diagnostic-only CUPTI; profiler changes scheduling; no speedup denominator",
           "queries": 32, "dim": 768, "batch": 1024, "workload": "random", "groups": {}}
for key, observations in sorted(groups.items()):
    summary["groups"][key] = {"count": len(observations), **{
        name: statistics.median(row[name] for row in observations) for name in observations[0]}}
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
