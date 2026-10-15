#!/usr/bin/env python3
"""Use CUPTI kernel intervals to verify overlap, never to claim a speedup."""
import json
import sqlite3
import sys
from pathlib import Path


def overlap(*intervals):
    return max(0, min(x[1] for x in intervals) - max(x[0] for x in intervals))


assert overlap((0, 4), (2, 5)) == 2
assert overlap((0, 2), (2, 5)) == 0


def inspect_trace(path):
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    rows = connection.execute("SELECT k.start,k.end,k.streamId,s.value "
                              "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                              "ON s.id=k.demangledName ORDER BY k.start").fetchall()
    queries = [x for x in rows if "sivf_search_kernel" in x[3]]
    results = []
    for q in queries:
        inserts = [x for x in rows if "sivf_append_kernel" in x[3] and x[2] != q[2] and overlap(q, x)]
        deletes = [x for x in rows if "sivf_delete_kernel" in x[3] and x[2] != q[2] and overlap(q, x)]
        results.append({"trial": len(results), "search_insert_overlap": bool(inserts),
                        "search_delete_overlap": bool(deletes),
                        "triple_overlap": any(overlap(q, a, b) for a in inserts for b in deletes)})
    connection.close()
    return {"scope": "diagnostic-only CUPTI kernel intervals; no latency or throughput claim",
            "search_kernels": len(queries),
            "search_insert_overlap_count": sum(x["search_insert_overlap"] for x in results),
            "search_delete_overlap_count": sum(x["search_delete_overlap"] for x in results),
            "triple_overlap_count": sum(x["triple_overlap"] for x in results), "trials": results}


if __name__ == "__main__":
    print(json.dumps(inspect_trace(sys.argv[1]), indent=2))
