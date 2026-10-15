#!/usr/bin/env python3
"""CPU count/retention screen. No GPU imports, timing, or concurrency claims."""
from __future__ import annotations
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import random
import sys

from blocks import Blocks, Oracles
from indexes import Index

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "reader-lifetime-20261015" / "model.py"
sys.path.insert(0, str(NATIVE.parent))
from model import Model as NativeDEBRModel

KINDS = ("chain", "avl", "array")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected(keys, timestamp):
    position = bisect_right(keys, timestamp) - 1
    return keys[position] if position >= 0 else None


def stats(values):
    ordered = sorted(values)
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[(95 * len(ordered) + 99) // 100 - 1], "max": ordered[-1]}


def lookup_screen():
    results, receipts = [], []
    for count in (16, 256, 4096):
        keys = list(range(0, count * 2, 2))
        rng = random.Random(17)
        queries = [-1, 0, count, count * 2 - 2, count * 2 - 1, count * 2]
        queries += [rng.randrange(-1, count * 2 + 1) for _ in range(1024)]
        for kind in KINDS:
            index = Index(kind)
            for key in keys:
                index.add(key)
            index.check()
            visits = []
            for query in queries:
                answer, inspected = index.predecessor(query)
                assert answer == expected(keys, query)
                visits.append(inspected)
                receipts.append([count, kind, query, answer, inspected])
            results.append({"versions": count, "index": kind,
                            "inspected_entries": stats(visits),
                            "oldest_query_visits": visits[1],
                            "newest_query_visits": visits[3],
                            "build_rotations": index.rotations,
                            "latest_direct_handle_index_visits": 0})
    return results, receipts


def history_pressure(kind, policy, budget=None, common_slots=None):
    entry_bytes = 512 + (32 if kind == "avl" else 16)
    descriptor_bytes = {"chain": 128, "avl": 256, "array": 512}[kind]
    slots = common_slots if common_slots else (budget - descriptor_bytes) // entry_bytes
    native = NativeDEBRModel(slots) if policy == "ebr" else None
    indexes = [Index(kind) for _ in range(16)]
    records = [{0: root} for root in range(16)]
    latest = [0] * 16
    for index in indexes:
        index.add(0)
    if native:
        for reader in range(8):
            native.acquire(reader)
    ack = failed = 0
    peak = 16
    first_fail = None
    trace = []

    def validate(held):
        physical = {slot for row in records for slot in row.values()}
        assert len(physical) == sum(map(len, records)) <= slots
        for root, index in enumerate(indexes):
            index.check()
            assert index.present == set(records[root])
            assert index.predecessor(513)[0] == latest[root]
            if held and root < 8:
                assert index.predecessor(0)[0] == 0, "snapshot visibility mismatch"
        if native:
            assert physical == native.used
            assert [records[r][latest[r]] for r in range(16)] == native.roots
            assert ack == native.ack and failed == native.failed
        return len(physical)

    for step in range(512):
        root, birth = step % 16, step + 1
        before = (latest[:], [dict(row) for row in records])
        if native:
            old_ack = native.ack
            native.update(step)
            slot = native.roots[root] if native.ack > old_ack else None
        else:
            free = set(range(slots)) - {slot for row in records for slot in row.values()}
            slot = min(free) if free else None
        if slot is None:
            failed += 1
            if first_fail is None:
                first_fail = step
            assert before == (latest, records)
        else:
            records[root][birth] = slot
            indexes[root].add(birth)
            latest[root] = birth
            ack += 1
            peak = max(peak, sum(map(len, records)))
            if policy == "precise":
                for old in list(records[root]):
                    if old != birth and not (root < 8 and old == 0):
                        indexes[root].remove(old)
                        del records[root][old]
        resident = validate(True)
        trace.append([step, int(slot is not None), resident, descriptor_bytes + resident * entry_bytes])
    before_release = validate(True)
    if native:
        for reader in range(8):
            native.release(reader)
        native.drain()
    for root in range(16):
        for old in list(records[root]):
            if old != latest[root]:
                indexes[root].remove(old)
                del records[root][old]
    assert validate(False) == 16
    return {"index": kind, "policy": policy, "byte_budget": budget,
            "common_entry_capacity": common_slots, "admitted_entry_capacity": slots,
            "entry_bytes": entry_bytes, "descriptor_bytes": descriptor_bytes,
            "reserved_layout_bytes": descriptor_bytes + slots * entry_bytes,
            "ack": ack, "failed": failed, "first_failed_attempt": first_fail,
            "peak_versions": peak, "peak_bytes": descriptor_bytes + peak * entry_bytes,
            "before_release_versions": before_release,
            "before_release_bytes": descriptor_bytes + before_release * entry_bytes,
            "final_versions": 16, "trace": trace}


def block_case(layout, blocks, policy, pattern):
    store = Blocks(layout, blocks, blocks * 4096 * 4, policy)
    oracles = Oracles(range(blocks * 8))
    rng = random.Random(23)
    trace = []
    for step in range(256):
        if step < 64 and step % 8 == 0:
            store.pins[step // 8] = store.root
            oracles[step // 8] = oracles.current
        block = 0 if pattern == "hot" else rng.randrange(blocks)
        key, value = block * 8 + rng.randrange(8), 1000000 + step
        unchanged = (store.root, store.serial, store.used, store.written)
        before_written, before_freed = store.written, store.freed
        success = store.replace(key, value)
        if success:
            values = list(oracles.current)
            values[key] = value
            oracles.current = tuple(values)
        else:
            assert unchanged == (store.root, store.serial, store.used, store.written)
        store.check(oracles, {0, blocks * 8 - 1, blocks * 4, key})
        trace.append([step, key, value, int(success), store.written - before_written,
                      store.freed - before_freed, store.used])
    store.check(oracles, range(blocks * 8))
    scan = store.scan_units()
    before_release = store.used
    store.pins.clear()
    oracles.clear()
    store.collect()
    store.check(oracles, range(blocks * 8))
    assert set(store.units) == store.reachable([store.root])
    assert store.used == store.initial_bytes
    return {"layout": layout, "blocks": blocks, "policy": policy, "pattern": pattern,
            "budget_bytes": store.budget, "initial_bytes": store.initial_bytes,
            "ack": store.ack, "failed": store.failed, "peak_bytes": store.peak,
            "before_release_bytes": before_release, "after_drain_bytes": store.used,
            "bytes_per_success": store.plan_bytes(), "written_bytes": store.written,
            "payload_written_bytes": store.payload_written,
            "metadata_written_bytes": store.metadata_written,
            "current_full_scan_units": scan,
            "final_values_sha256": hashlib.sha256(json.dumps(oracles.current).encode()).hexdigest(),
            "trace": trace}


def self_tests():
    for kind in KINDS:
        for seed in range(100):
            rng, index, keys = random.Random(seed), Index(kind), set()
            for _ in range(500):
                key, event = rng.randrange(128), rng.randrange(3)
                if event == 0 and key not in keys:
                    index.add(key)
                    keys.add(key)
                elif event == 1 and key in keys:
                    index.remove(key)
                    keys.remove(key)
                query = rng.randrange(-1, 257)
                assert index.predecessor(query)[0] == expected(sorted(keys), query)
                index.check()
        mutant = Index(kind)
        for key in (0, 2, 4):
            mutant.add(key)
        mutant.remove(2)
        try:
            assert mutant.predecessor(3)[0] == 2, "snapshot visibility mismatch"
        except AssertionError as error:
            assert str(error) == "snapshot visibility mismatch"
        else:
            raise AssertionError("lost snapshot version was accepted")
    for layout in ("whole", "flat", "tree"):
        for policy in ("hold_no_reclaim", "precise"):
            for seed in range(20):
                rng = random.Random(seed)
                store = Blocks(layout, 16, 64 * 1024 * 1024, policy)
                oracles = Oracles(range(128))
                for step in range(64):
                    reader, event = rng.randrange(8), rng.randrange(3)
                    if event == 0 and reader not in store.pins:
                        store.pins[reader] = store.root
                        oracles[reader] = oracles.current
                    elif event == 1 and reader in store.pins:
                        del store.pins[reader], oracles[reader]
                        if policy == "precise":
                            store.collect()
                    else:
                        key = rng.randrange(128)
                        assert store.replace(key, 1000000 + step)
                        values = list(oracles.current)
                        values[key] = 1000000 + step
                        oracles.current = tuple(values)
                    store.check(oracles, range(128))
                store.pins.clear()
                oracles.clear()
                store.collect()
                store.check(oracles, range(128))
                assert store.used == store.initial_bytes
            store = Blocks(layout, 16, 4 * 16 * 4096, policy)
            store.budget = store.used
            before = (store.root, store.serial, store.used, dict(store.units))
            assert not store.replace(0, -1)
            assert before == (store.root, store.serial, store.used, store.units)
    return {"index_seeds": 100, "index_events_per_seed": 500,
            "block_seeds": 20, "block_events_per_seed": 64,
            "visibility_negative_mutations_rejected": list(KINDS),
            "failed_admission_no_mutation": True, "passed": True}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    validation = self_tests()
    lookups, receipts = lookup_screen()
    pressure = [history_pressure(kind, policy, budget, capacity)
                for budget, capacity in ((None, 64), (65536, None), (262144, None))
                for policy in ("ebr", "precise") for kind in KINDS]
    block = [block_case(layout, count, policy, pattern)
             for count in (16, 256, 4096) for pattern in ("uniform", "hot")
             for policy in ("hold_no_reclaim", "precise") for layout in ("whole", "flat", "tree")]
    with (out / "LOOKUP.jsonl").open("w") as stream:
        for row in receipts:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    sources = {name: digest(HERE / name) for name in
               ("indexes.py", "blocks.py", "check.py", "CONTRACT.md", "AMENDMENT.md")}
    sources["../reader-lifetime-20261015/model.py"] = digest(NATIVE)
    result = {"experiment": "ivf_20261015_version_tree_sc", "sources": sources,
              "scope": "serialized CPU count model; no GPU/concurrent protocol or new mechanism",
              "validation": validation, "lookups": lookups, "pressure": pressure,
              "blocks": block, "lookup_receipts_sha256": digest(out / "LOOKUP.jsonl")}
    (out / "RESULTS.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    for row in lookups:
        print("LOOKUP", row["versions"], row["index"], row["inspected_entries"])
    for row in pressure:
        print("PRESSURE", row["byte_budget"], row["common_entry_capacity"], row["index"],
              row["policy"], "ack", row["ack"], "peak bytes", row["peak_bytes"])
    for row in block:
        print("BLOCK", row["blocks"], row["pattern"], row["policy"], row["layout"],
              "ack", row["ack"], "copy/success", row["bytes_per_success"],
              "resident", row["before_release_bytes"])
    print("PASS: bounded CPU model checks only; GPU gates remain pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.out:
        run(args.out)
    elif args.self_test:
        print(self_tests())
    else:
        parser.error("use --out NEW_DIRECTORY or --self-test")
