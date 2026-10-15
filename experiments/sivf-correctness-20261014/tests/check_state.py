#!/usr/bin/env python3
"""Check quiescent upstream SIVF snapshots against the logical CPU oracle."""
import argparse
import json
import math
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(f"Malformed evidence: {message}")


def validate_snapshot(state):
    pool, dim = state["pool_size"], state["dim"]
    require(type(pool) is int and pool > 0 and type(dim) is int and dim > 0, "pool/dimension")
    require(len(state["slabs"]) == pool and len(state["heads"]) > 0, "slab/head array sizes")
    require(all(type(x) is int for x in state["heads"] + state["free_list"]), "slab index types")
    require(type(state["free_list_top"]) is int, "free stack top type")
    att = [int(x) for x in state["address_table"]]
    require(all(0 <= x < 1 << 64 for x in att), "ATT value range")
    oracle_ids = [x["id"] for x in state["oracle"]]
    require(len(oracle_ids) == len(set(oracle_ids)), "duplicate oracle IDs")
    for entry in state["oracle"]:
        require(type(entry["id"]) is int and 0 <= entry["id"] < len(att), "oracle ID range")
        require(type(entry["list"]) is int and 0 <= entry["list"] < len(state["heads"]), "oracle owner")
        require(len(entry["vector"]) == dim and all(type(x) in (int, float) and math.isfinite(x) for x in entry["vector"]), "oracle payload")
    for index, slab in enumerate(state["slabs"]):
        require(slab["index"] == index and type(slab["next"]) is int, "slab index/next")
        require(type(slab["count"]) is int and type(slab["bitmap"]) is int and 0 <= slab["bitmap"] < 1 << 32, "slab count/bitmap")
        slots = [x["slot"] for x in slab["records"]]
        require(all(type(x) is int and 0 <= x < 32 for x in slots) and len(set(slots)) == len(slots), "physical slots")
        require(sum(1 << x for x in slots) == slab["bitmap"], "records do not cover the valid bitmap")
        for record in slab["records"]:
            require(type(record["id"]) is int and len(record["vector"]) == dim, "physical ID/payload shape")


def inspect_search(state):
    labels, distances, reference = state["labels"], state["distances"], state["reference"]
    require(len(labels) == len(distances) == 10, "top-10 output arrays")
    require(all(type(x) is int for x in labels), "search label types")
    require(all(type(x) in (int, float) and math.isfinite(x) for x in distances), "search distances")
    expected = [x["id"] for x in reference]
    require(len(expected) <= 10 and len(set(expected)) == len(expected), "reference ID count")
    require(all(type(x["id"]) is int and type(x["distance"]) in (int, float) and math.isfinite(x["distance"]) and x["distance"] >= 0 for x in reference), "search reference")
    passed = labels == expected + [-1] * (10 - len(expected))
    passed &= all(math.isclose(distances[i], row["distance"], rel_tol=1e-4, abs_tol=1e-4)
                  for i, row in enumerate(reference))
    return {"passed": passed, "actual": labels, "expected": expected}


def inspect(state):
    validate_snapshot(state)
    errors = []
    pool = state["pool_size"]
    slabs = state["slabs"]
    free = state["free_list"]
    att = [int(x) for x in state["address_table"]]
    oracle = {x["id"]: x for x in state["oracle"]}
    owners = {}
    records = {}
    if state["free_list_top"] != len(free) or len(set(free)) != len(free):
        errors.append("I4: invalid free stack size or duplicate free slab")
    if any(not 0 <= x < pool for x in free):
        errors.append("I4: free slab index out of range")
    for owner, head in enumerate(state["heads"]):
        seen = set()
        while head != -1:
            if not 0 <= head < pool:
                errors.append(f"I3: list {owner} points outside pool: {head}")
                break
            if head in seen:
                errors.append(f"I3: list {owner} has a cycle at slab {head}")
                break
            seen.add(head)
            if head in owners:
                errors.append(f"I3: slab {head} belongs to lists {owners[head]} and {owner}")
            else:
                owners[head] = owner
            if head in free:
                errors.append(f"I4: reachable slab {head} is free")
            slab = slabs[head]
            if slab["count"] != bin(slab["bitmap"]).count("1"):
                errors.append(f"I5: slab {head}: count={slab['count']} popcount={bin(slab['bitmap']).count('1')}")
            for record in slab["records"]:
                uid, slot = record["id"], record["slot"]
                if uid in records:
                    errors.append(f"I1: duplicate reachable ID {uid}")
                records[uid] = (head, slot)
                if uid not in oracle:
                    errors.append(f"I1: reachable ID {uid} is absent from oracle")
                    continue
                if oracle[uid]["list"] != owner:
                    errors.append(f"I3: ID {uid} is in wrong list {owner}")
                if not 0 <= uid < len(att) or att[uid] != (head << 32) | slot:
                    errors.append(f"I1: reverse mapping differs from ATT for ID {uid}")
                if record["vector"] != oracle[uid]["vector"]:
                    errors.append(f"I2: payload differs from oracle for ID {uid}")
            head = slab["next"]
    coords = {}
    for uid in oracle:
        coordinate = att[uid]
        if uid not in records:
            errors.append(f"I1/I2: expected live ID {uid} is missing")
        if coordinate in coords:
            errors.append(f"I2: active IDs {coords[coordinate]} and {uid} share ATT {coordinate}")
        coords[coordinate] = uid
        if coordinate == (1 << 64) - 1:
            errors.append(f"I1: expected live ID {uid} has invalid ATT")
    orphans = set(range(pool)) - set(free) - set(owners)
    if orphans:
        errors.append(f"RESOURCE: {len(orphans)} slabs are neither reachable nor free")
    return {"passed": not errors, "errors": errors, "reachable": len(owners),
            "free": len(free), "orphan": len(orphans), "oracle_live": len(oracle),
            "reachable_ids": len(records)}


def self_test():
    clean = {"dim": 1, "pool_size": 1, "heads": [0], "free_list_top": 0, "free_list": [],
             "address_table": ["0"], "oracle": [{"id": 0, "list": 0, "vector": [2.0]}],
             "slabs": [{"index": 0, "next": -1, "count": 1, "bitmap": 1,
                        "records": [{"id": 0, "slot": 0, "vector": [2.0]}]}]}
    assert inspect(clean)["passed"]
    for field, replacement in [("free_list", [0]), ("oracle", []), ("address_table", ["1"])]:
        changed = dict(clean, **{field: replacement})
        assert not inspect(changed)["passed"], field
    damaged = json.loads(json.dumps(clean))
    damaged["slabs"][0]["next"] = 0
    assert any("cycle" in x for x in inspect(damaged)["errors"])
    damaged["slabs"][0]["next"] = -1
    damaged["slabs"][0]["count"] = 2
    assert any("I5" in x for x in inspect(damaged)["errors"])
    print("checker self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    failed = False
    for name in args.paths:
        path = Path(name)
        result = inspect(json.loads(path.read_text()))
        failed |= not result["passed"]
        print(json.dumps({"snapshot": str(path), **result}))
    sys.exit(1 if failed else 0)
