"""Immutable block-storage models with explicit packed-layout byte admission."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    kind: str
    data: tuple
    children: tuple
    size: int


class Blocks:
    def __init__(self, layout, blocks, budget, policy):
        assert layout in ("whole", "flat", "tree")
        assert policy in ("hold_no_reclaim", "precise")
        assert blocks > 0 and blocks & (blocks - 1) == 0
        self.layout, self.blocks, self.budget, self.policy = layout, blocks, budget, policy
        self.units, self.pins = {}, {}
        self.descriptor_bytes = 72  # Current root plus eight reserved snapshot slots.
        self.serial = self.written = self.freed = 0
        self.used = self.peak = self.descriptor_bytes
        self.payload_written = self.metadata_written = self.ack = self.failed = 0
        if layout == "whole":
            self.root = self.new("whole", tuple(range(blocks * 8)), (), blocks * 4096 + 16)
        else:
            leaves = [self.new("leaf", tuple(range(i * 8, (i + 1) * 8)), (), 4112)
                      for i in range(blocks)]
            if layout == "flat":
                self.root = self.new("directory", (), tuple(leaves), 16 + 8 * blocks)
            else:
                def build(ids):
                    if len(ids) == 1:
                        return ids[0]
                    half = len(ids) // 2
                    return self.new("branch", (), (build(ids[:half]), build(ids[half:])), 32)
                self.root = build(leaves)
        self.initial_bytes = self.used
        self.written = self.payload_written = self.metadata_written = 0

    def new(self, kind, data, children, size):
        assert self.used + size <= self.budget
        uid = self.serial
        self.serial += 1
        self.units[uid] = Unit(kind, data, children, size)
        self.used += size
        self.peak = max(self.peak, self.used)
        payload = self.blocks * 4096 if kind == "whole" else 4096 if kind == "leaf" else 0
        self.written += size
        self.payload_written += payload
        self.metadata_written += size - payload
        return uid

    def read(self, root, key):
        assert 0 <= key < self.blocks * 8
        uid, block = root, key // 8
        unit = self.units[uid]
        if unit.kind == "whole":
            return unit.data[key]
        if unit.kind == "directory":
            return self.units[unit.children[block]].data[key % 8]
        low, high = 0, self.blocks
        while unit.kind == "branch":
            mid = (low + high) // 2
            right = block >= mid
            uid = unit.children[int(right)]
            low, high = (mid, high) if right else (low, mid)
            unit = self.units[uid]
        return unit.data[key % 8]

    def plan_bytes(self):
        if self.layout == "whole":
            return self.blocks * 4096 + 16
        if self.layout == "flat":
            return 4112 + 16 + 8 * self.blocks
        return 4112 + (self.blocks.bit_length() - 1) * 32

    def replace(self, key, value):
        planned = self.plan_bytes()
        if self.used + planned > self.budget:
            self.failed += 1
            return False
        before = self.written
        old = self.units[self.root]
        if self.layout == "whole":
            values = list(old.data)
            values[key] = value
            new_root = self.new("whole", tuple(values), (), self.blocks * 4096 + 16)
        elif self.layout == "flat":
            block = key // 8
            children = list(old.children)
            values = list(self.units[children[block]].data)
            values[key % 8] = value
            children[block] = self.new("leaf", tuple(values), (), 4112)
            new_root = self.new("directory", (), tuple(children), 16 + 8 * self.blocks)
        else:
            def copy_path(uid, low, high):
                unit = self.units[uid]
                if unit.kind == "leaf":
                    values = list(unit.data)
                    values[key % 8] = value
                    return self.new("leaf", tuple(values), (), 4112)
                mid = (low + high) // 2
                side = int(key // 8 >= mid)
                children = list(unit.children)
                bounds = (mid, high) if side else (low, mid)
                children[side] = copy_path(children[side], *bounds)
                return self.new("branch", (), tuple(children), 32)
            new_root = copy_path(self.root, 0, self.blocks)
        assert self.written - before == planned
        self.root = new_root
        self.ack += 1
        if self.policy == "precise":
            self.collect()
        return True

    def reachable(self, roots):
        seen, todo = set(), list(roots)
        while todo:
            uid = todo.pop()
            assert uid in self.units, "dangling storage unit"
            if uid not in seen:
                seen.add(uid)
                todo.extend(self.units[uid].children)
        return seen

    def collect(self):
        keep = self.reachable([self.root] + list(self.pins.values()))
        dead = set(self.units) - keep
        size = sum(self.units[uid].size for uid in dead)
        for uid in dead:
            del self.units[uid]
        self.used -= size
        self.freed += size

    def check(self, oracles, keys):
        assert self.used == self.descriptor_bytes + sum(unit.size for unit in self.units.values()) <= self.budget
        assert self.initial_bytes + self.written == self.used + self.freed
        assert set(oracles) == set(self.pins)
        for root, values in [(self.root, oracles.current)] + [
                (self.pins[reader], values) for reader, values in oracles.items()]:
            self.reachable([root])
            for key in keys:
                assert self.read(root, key) == values[key], "snapshot storage mismatch"

    def scan_units(self):
        ids = self.reachable([self.root])
        return {"payload_units": sum(self.units[i].kind in ("leaf", "whole") for i in ids),
                "directory_units": sum(self.units[i].kind in ("directory", "branch") for i in ids),
                "directory_bytes": sum(self.units[i].size for i in ids
                                       if self.units[i].kind in ("directory", "branch"))}


class Oracles(dict):
    def __init__(self, values):
        super().__init__()
        self.current = tuple(values)
