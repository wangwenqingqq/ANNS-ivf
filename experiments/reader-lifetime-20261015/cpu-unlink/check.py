#!/usr/bin/env python3
"""CPU-only topology/retention ablations and bounded SC counterexamples."""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model import Model as NativeDEBRModel


@dataclass
class Node:
    slot: int
    root: int
    birth: int
    older: int | None = None
    newer: int | None = None


class Store:
    def __init__(self, slots, linkage, policy):
        assert linkage in ("S", "D") and policy in ("ebr", "precise")
        self.slots, self.linkage, self.policy = slots, linkage, policy
        self.native = NativeDEBRModel(slots) if policy == "ebr" else None
        self.nodes = {i: Node(i, i, 0) for i in range(16)}
        self.heads = list(range(16))
        self.linked = set(self.nodes)
        self.queries, self.cursors = {}, {}
        self.clock = self.ack = self.failed = self.freed = 0
        self.peak = 16
        self.first_fail = None
        self.classified = self.forward_writes = self.back_writes = 0
        self.trace = []

    def walk(self, uid):
        seen = []
        while uid is not None:
            assert uid in self.nodes, "dangling version reference"
            assert uid not in seen, "cycle"
            seen.append(uid)
            uid = self.nodes[uid].older
        return seen

    def find(self, uid, timestamp):
        for candidate in self.walk(uid):
            if self.nodes[candidate].birth <= timestamp:
                return candidate
        raise AssertionError("snapshot version missing")

    def begin(self, reader):
        assert reader not in self.queries and 0 <= reader < 8
        self.queries[reader] = (reader, self.clock, self.heads[reader])
        if self.native:
            self.native.acquire(reader)
        self.check()

    def start_cursor(self, reader):
        assert reader in self.queries and reader not in self.cursors
        self.cursors[reader] = self.heads[self.queries[reader][0]]
        self.check()

    def advance_cursor(self, reader):
        uid = self.cursors[reader]
        _, timestamp, expected = self.queries[reader]
        if self.nodes[uid].birth > timestamp:
            self.cursors[reader] = self.nodes[uid].older
        else:
            assert uid == expected
        self.gc()
        self.check()

    def end(self, reader):
        self.cursors.pop(reader, None)
        del self.queries[reader]
        if self.native:
            self.native.release(reader)
        self.gc()
        self.check()

    def replace(self, root, collect=True):
        step = self.clock
        self.clock += 1
        if self.native:
            # This adapter intentionally supports only the preregistered all-held trace.
            assert len(self.queries) == 8 and all(q[1] == 0 for q in self.queries.values())
            assert root == step % 16 and not self.cursors
            old_ack = self.native.ack
            self.native.update(step)
            slot = self.native.roots[root] if self.native.ack > old_ack else None
        else:
            free = set(range(self.slots)) - {node.slot for node in self.nodes.values()}
            slot = min(free) if free else None
        if slot is None:
            self.failed += 1
            if self.first_fail is None:
                self.first_fail = step
        else:
            uid, previous = 16 + step, self.heads[root]
            self.nodes[uid] = Node(slot, root, self.clock, previous)
            if self.linkage == "D":
                self.nodes[previous].newer = uid
                self.back_writes += 1
            self.heads[root] = uid
            self.linked.add(uid)
            self.ack += 1
            self.peak = max(self.peak, len(self.nodes))
            if collect:
                self.gc()
        self.check()
        self.trace.append([step, int(slot is not None), len(self.nodes), len(self.linked)])

    def gc(self):
        needed = set(self.heads) | {query[2] for query in self.queries.values()}
        # Both topologies classify each current chain once. S keeps its predecessor.
        for head in self.heads:
            previous = None
            for uid in self.walk(head):
                self.classified += 1
                if uid in needed:
                    previous = uid
                    continue
                node = self.nodes[uid]
                newer = node.newer if self.linkage == "D" else previous
                assert newer is not None and newer in self.linked
                assert self.nodes[newer].older == uid
                self.nodes[newer].older = node.older
                self.forward_writes += 1
                if self.linkage == "D" and node.older is not None:
                    self.nodes[node.older].newer = newer
                    self.back_writes += 1
                self.linked.remove(uid)
                # Do not clear a detached node's old traversal link: a cursor may own it.
        if self.policy == "precise":
            retained = set(self.linked)
            for cursor in self.cursors.values():
                retained.update(self.walk(cursor))
            released = set(self.nodes) - retained
            for uid in released:
                del self.nodes[uid]
            self.freed += len(released)

    def check(self):
        physical = {node.slot for node in self.nodes.values()}
        assert len(physical) == len(self.nodes) <= self.slots
        assert self.freed + len(self.nodes) == 16 + self.ack
        reachable = []
        for root, head in enumerate(self.heads):
            chain = self.walk(head)
            assert all(self.nodes[uid].root == root for uid in chain)
            if self.linkage == "D":
                for newer, older in zip(chain, chain[1:]):
                    assert self.nodes[older].newer == newer
                assert self.nodes[head].newer is None
            reachable.extend(chain)
        assert len(set(reachable)) == len(reachable)
        assert set(reachable) == self.linked
        for reader, (root, timestamp, expected) in self.queries.items():
            assert self.find(self.heads[root], timestamp) == expected
            if reader in self.cursors:
                assert self.find(self.cursors[reader], timestamp) == expected
        if self.native:
            assert physical == self.native.used
            assert [self.nodes[h].slot for h in self.heads] == self.native.roots
            assert self.ack == self.native.ack and self.failed == self.native.failed

    def drain(self):
        assert not self.queries and not self.cursors
        self.gc()
        if self.native:
            self.native.drain()
            released = set(self.nodes) - set(self.heads)
            for uid in released:
                del self.nodes[uid]
            self.freed += len(released)
        self.check()
        assert len(self.nodes) == 16 and self.freed == self.ack


def ablation(slots, linkage, policy):
    store = Store(slots, linkage, policy)
    for reader in range(8):
        store.begin(reader)
    for step in range(512):
        store.replace(step % 16)
    before = {"resident": len(store.nodes), "linked": len(store.linked),
              "unlinked_resident": len(store.nodes) - len(store.linked)}
    for reader in range(8):
        store.end(reader)
    store.drain()
    return {"slots": slots, "linkage": linkage, "retention": policy,
            "ack": store.ack, "failed": store.failed, "first_fail": store.first_fail,
            "peak_slots": store.peak, "before_release": before,
            "final_live": len(store.nodes), "classified_nodes": store.classified,
            "forward_rewires": store.forward_writes, "back_link_rewires": store.back_writes,
            "extra_back_link_fields_per_reserved_slot": int(linkage == "D"),
            "trace_columns": ["attempt_zero_based", "acknowledged", "resident", "linked"],
            "trace": store.trace}


def naive_removals(schedule):
    links = {"A": [None,"B"], "B": ["A","C"], "C": ["B","D"], "D": ["C",None]}
    pc, saved = {"B":0,"C":0}, {"B":{},"C":{}}
    for actor in schedule:
        step = pc[actor]
        if step == 0:
            saved[actor]["previous"] = links[actor][0]
        elif step == 1:
            saved[actor]["next"] = links[actor][1]
        elif step == 2:
            links[saved[actor]["previous"]][1] = saved[actor]["next"]
        else:
            links[saved[actor]["next"]][0] = saved[actor]["previous"]
        pc[actor] += 1
    valid = links["A"][1] == "D" and links["D"][0] == "A"
    return {"schedule": schedule, "valid_remaining_endpoints": valid, "final_links": links}


def adjacent_schedules():
    results = []
    for positions in itertools.combinations(range(8),4):
        schedule = "".join("B" if i in positions else "C" for i in range(8))
        results.append(naive_removals(schedule))
    assert len(results) == 70
    assert naive_removals("BBBBCCCC")["valid_remaining_endpoints"]
    assert naive_removals("CCCCBBBB")["valid_remaining_endpoints"]
    invalid = [row for row in results if not row["valid_remaining_endpoints"]]
    assert invalid
    return {"total_schedules": 70, "broken_schedules": len(invalid),
            "scope": "two naive four-step removals; no freeing or weak memory",
            "first_witness": invalid[0], "all_schedules": results}


def cursor_counterexample(linkage):
    store = Store(64, linkage, "precise")
    store.begin(0)
    store.replace(0, collect=False)  # v1 = 16
    store.replace(0, collect=False)  # v2 = 17
    store.start_cursor(0)
    store.replace(0, collect=False)  # v3 = 18
    store.gc()
    store.check()
    assert store.walk(store.heads[0]) == [18,0]
    assert store.walk(store.cursors[0]) == [17,16,0]
    mutant = copy.deepcopy(store)
    del mutant.nodes[16]  # Unneeded by snapshots and not directly announced, but still transit-reachable.
    mutant.freed += 1  # Keep accounting consistent, so rejection must localize to the stale cursor path.
    try:
        mutant.check()
    except AssertionError as error:
        assert str(error) == "dangling version reference"
    else:
        raise AssertionError("direct-cursor-only mutation escaped")
    before = len(store.nodes)
    store.advance_cursor(0)
    store.advance_cursor(0)
    assert store.cursors[0] == 0 and 16 not in store.nodes and 17 not in store.nodes
    store.end(0)
    store.drain()
    return {"linkage": linkage, "root_chain_after_unlink": [18,0],
            "held_cursor_path_after_unlink": [17,16,0], "unsafe_free_candidate":16,
            "direct_cursor_only_mutation_rejected":True,
            "mutation_rejection_reason":"dangling version reference",
            "resident_before_cursor_progress":before, "final_live":len(store.nodes)}


def random_checks():
    for linkage in ("S","D"):
        for seed in range(100):
            rng = random.Random(seed)
            store = Store(64,linkage,"precise")
            for _ in range(500):
                reader = rng.randrange(8)
                event = rng.randrange(5)
                if event == 0 and reader not in store.queries:
                    store.begin(reader)
                elif event == 1 and reader in store.queries:
                    store.end(reader)
                elif event == 2 and reader in store.queries:
                    if reader in store.cursors:
                        store.advance_cursor(reader)
                    else:
                        store.start_cursor(reader)
                else:
                    store.replace(rng.randrange(16))
                store.check()
            for reader in list(store.queries):
                store.end(reader)
            store.drain()
    return {"topologies":2,"seeds_per_topology":100,"events_per_seed":500,"passed":True}


def run():
    cases = [ablation(slots,link,policy) for slots in (64,256)
             for policy in ("ebr","precise") for link in ("S","D")]
    for slots in (64,256):
        rows = [r for r in cases if r["slots"]==slots]
        for policy in ("ebr","precise"):
            s,d = [r for r in rows if r["retention"]==policy]
            for field in ("ack","failed","first_fail","peak_slots","before_release","final_live","classified_nodes","forward_rewires","trace"):
                assert s[field] == d[field], (policy,field)
        for r in rows:
            assert r["final_live"]==16
            if r["retention"]=="ebr":
                assert r["first_fail"]==slots-16 and r["ack"]==slots-16
            else:
                assert r["ack"]==512 and r["peak_slots"]==25
    return {"experiment":"ivf_20261015_unlink_retention_sc_model",
            "kind":"CPU SC model only; serialized oracle-assisted GC; no GPU execution",
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_transition_model_sha256":hashlib.sha256((HERE.parent/"model.py").read_bytes()).hexdigest(),
            "cases":cases,"adjacent_removals":adjacent_schedules(),
            "cursor_counterexamples":[cursor_counterexample(link) for link in ("S","D")],
            "random_checks":random_checks(),"gpu_results":False,"novel_mechanism_admitted":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=run()
    if args.out:
        args.out.write_text(json.dumps(result,indent=2)+"\n")
    for r in result["cases"]:
        print(r["slots"],r["retention"],r["linkage"],"ack",r["ack"],"peak",r["peak_slots"],"resident/linked",r["before_release"])
    print("naive adjacent-removal schedules broken:",result["adjacent_removals"]["broken_schedules"],"/ 70")
    print("PASS: bounded CPU-only checks; not a GPU or concurrent-protocol proof")
