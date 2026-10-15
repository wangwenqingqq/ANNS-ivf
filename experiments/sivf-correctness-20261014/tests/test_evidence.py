#!/usr/bin/env python3
"""Regression checks for malformed snapshots, truncated searches, and missing runs."""
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

from check_state import inspect, inspect_search


def rejects(function, value):
    try:
        function(value)
    except (ValueError, KeyError, TypeError, IndexError):
        return
    raise AssertionError("malformed evidence was accepted")


clean = {"dim": 1, "pool_size": 1, "heads": [0], "free_list_top": 0, "free_list": [],
         "address_table": ["0"], "oracle": [{"id": 0, "list": 0, "vector": [2.0]}],
         "slabs": [{"index": 0, "next": -1, "count": 1, "bitmap": 1,
                    "records": [{"id": 0, "slot": 0, "vector": [2.0]}]}]}
assert inspect(clean)["passed"]
for field, replacement in (("dim", 2), ("slabs", []), ("address_table", []),
                           ("oracle", clean["oracle"] * 2)):
    rejects(inspect, dict(clean, **{field: replacement}))
for records in ([], clean["slabs"][0]["records"] * 2):
    damaged = copy.deepcopy(clean)
    damaged["slabs"][0]["records"] = records
    rejects(inspect, damaged)

search = {"labels": [0] + [-1] * 9, "distances": [0.0] * 10,
          "reference": [{"id": 0, "distance": 0.0}]}
assert inspect_search(search)["passed"]
assert not inspect_search(dict(search, labels=[1] + [-1] * 9))["passed"]
for field, replacement in (("labels", []), ("distances", []), ("distances", [float("nan")] * 10)):
    rejects(inspect_search, dict(search, **{field: replacement}))

with tempfile.TemporaryDirectory() as directory:
    Path(directory, "SOURCE_COMMIT.txt").write_text("test\n")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), directory],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "missing required artifacts" in result.stderr
    assert not Path(directory, "EVIDENCE_SUMMARY.json").exists()
print("evidence regression tests: PASS")
