#!/usr/bin/env python3
"""State-transition model of the pinned native DEBR; not GPU timing or a proof."""
import argparse
import json
from pathlib import Path
import random


class Model:
    def __init__(self, slots=64):
        self.slots = slots
        self.epoch = 0
        self.announce = [1] * 9
        self.bag = [0] * 9
        self.bags = [[set(), set(), set()] for _ in range(9)]
        self.roots = list(range(16))
        self.used = set(self.roots)
        self.pins = {}
        self.peak = 16
        self.ack = self.failed = self.freed = 0
        self.first_fail = None

    def leave(self, actor):
        current, old = self.epoch, self.announce[actor] & ~1
        self.announce[actor] = current
        if old != current:
            self.bag[actor] = (self.bag[actor] + 1) % 3
            released = self.bags[actor][self.bag[actor]]
            assert not released.intersection(self.pins.values()), "early free"
            assert not released.intersection(self.roots), "free live root"
            self.used.difference_update(released)
            self.freed += len(released)
            released.clear()
        if all(value & 1 or (value & ~1) == current for value in self.announce):
            self.epoch = current + 2

    def enter(self, actor):
        self.announce[actor] |= 1

    def acquire(self, actor):
        assert actor not in self.pins
        self.leave(actor)
        self.pins[actor] = self.roots[actor]

    def release(self, actor):
        del self.pins[actor]
        self.enter(actor)

    def update(self, step):
        self.leave(8)
        available = set(range(self.slots)) - self.used
        if not available:
            self.failed += 1
            if self.first_fail is None:
                self.first_fail = step
        else:
            slot = min(available)
            self.used.add(slot)
            self.peak = max(self.peak, len(self.used))
            old, self.roots[step % 16] = self.roots[step % 16], slot
            self.bags[8][self.bag[8]].add(old)
            self.ack += 1
        self.enter(8)
        self.check()

    def check(self):
        retired = [v for actor in self.bags for bag in actor for v in bag]
        assert len(set(self.roots)) == 16
        assert len(retired) == len(set(retired))
        assert set(self.roots).isdisjoint(retired)
        assert self.used == set(self.roots) | set(retired)
        assert set(self.pins.values()) <= self.used
        assert len(self.used) <= self.slots
        assert self.freed + len(retired) == self.ack

    def drain(self):
        assert not self.pins
        for _ in range(16):
            self.leave(8)
            self.enter(8)
        self.check()
        assert len(self.used) == 16 and self.freed == self.ack


def scenario(slots, scope, name):
    m = Model(slots)
    # Event-count schedule, intentionally not a conversion of 2000 us to updates.
    if scope == "E" or name == "inside":
        for reader in range(8):
            m.acquire(reader)
    for step in range(512):
        m.update(step)
    if not m.pins:
        for reader in range(8):
            m.acquire(reader)
    protected = len(set(m.roots) | set(m.pins.values()))
    for reader in range(8):
        m.release(reader)
    m.drain()
    return {"slots": slots, "scope": scope, "scenario": name, "attempts": 512,
            "ack": m.ack, "failed": m.failed, "first_fail": m.first_fail,
            "peak_slots": m.peak, "needed_at_end_of_hold": protected,
            "final_live": len(m.used)}


def self_test():
    for slots in (64, 256):
        for scope in "EL":
            r = scenario(slots, scope, "inside")
            assert r["first_fail"] == slots - 16 and r["peak_slots"] == slots
        early, late = [scenario(slots, s, "pre") for s in "EL"]
        assert early["ack"] == slots - 16 and late["ack"] == 512
        assert late["peak_slots"] == 19 and late["failed"] == 0
    # Random legal reader starts/ends interleaved with all three bag rotations.
    for seed in range(100):
        rng = random.Random(seed)
        m = Model(64)
        for step in range(2000):
            reader = rng.randrange(8)
            if rng.random() < .2:
                m.release(reader) if reader in m.pins else m.acquire(reader)
            m.update(step)
        for reader in list(m.pins):
            m.release(reader)
        m.drain()
    # Verify the observer catches a known premature-free mutation.
    m = Model();m.acquire(0);m.update(0);m.used.remove(m.pins[0])
    try:
        m.check()
    except AssertionError:
        pass
    else:
        raise AssertionError("oracle accepted a lost pinned version")
    print("DEBR state-transition/model oracle tests: PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.self_test:
        self_test()
    if args.out:
        args.out.write_text(json.dumps({"kind": "CPU discrete-event model, not measured GPU",
            "upstream_commit": "f755006d78ccca565f97cda053727d4d48b82d46",
            "schedule": "512 writer attempts wholly inside pre/inside hold; no time conversion",
            "cases": [scenario(slots, scope, name) for slots in (64,256)
                      for name in ("pre","inside") for scope in "EL"]}, indent=2)+"\n")
