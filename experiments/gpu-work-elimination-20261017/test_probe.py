"""Regression gates and independent work-receipt checks, no GPU access."""
import json
from pathlib import Path
import unittest
import numpy as np
import probe


class FactorTests(unittest.TestCase):
    def test_primitive(self):
        self.assertEqual(probe.self_test()["exact_integer_cases"], 144)

    def test_repeated_query_positive_control(self):
        row = np.array([0xA1B2C3D4, 0x76543210, 0xCAFEBABE, 0x13579BDF], dtype=np.uint32)
        plans = probe.dictionary(np.tile(row, (32, 1)), False)
        self.assertEqual(len(plans[0]), 4)
        self.assertEqual(len(plans[0]) / (4 * 32), 1 / 32)
        words = np.array([0, 1, probe.FULL, 0x12345678], dtype=np.uint32)
        actual = probe.apply_plan(words, plans)
        expected = np.array([sum(weight * sum(((int(x) >> b) & 1) * ((int(mask) >> b) & 1) for b in range(32)) for mask, weight in zip(row, (1, 2, 4, -8))) for x in words])
        self.assertTrue(np.array_equal(actual, np.repeat(expected[:, None], 32, axis=1)))

    def test_complement_can_cost_more(self):
        masks = np.array([[1, 2, 4, 0x80000000]], dtype=np.uint32)
        ordinary = probe.dictionary(masks, False)
        complement = probe.dictionary(masks, True)
        self.assertEqual(len(ordinary[0]), 4)
        self.assertEqual(len(complement[0]) + int(np.any(complement[2])), 5)

    def test_zero_and_ones(self):
        for raw in (0, probe.FULL):
            plans = probe.dictionary(np.full((32, 4), raw, dtype=np.uint32), True)
            words = np.array([0, 1, probe.FULL], np.uint32)
            expected = np.array([0 if raw == 0 else -int(x).bit_count() for x in words])
            self.assertTrue(np.array_equal(probe.apply_plan(words, plans), np.repeat(expected[:, None], 32, axis=1)))

    def test_centroid_scale_is_not_a_bound(self):
        distances = np.square(np.array([-1., 1.]) - 0.)
        centroid_heuristic = 1.45 * 0.
        self.assertLess(min(distances[0], centroid_heuristic), distances.min())

    def test_receipts_when_available(self):
        root = Path(__file__).resolve().parent
        if not (root / "RESULTS.json").exists():
            self.skipTest("copy retained receipts to test aggregate consistency")
        import gzip
        result = json.loads((root / "RESULTS.json").read_text())
        with gzip.open(root / "GROUPS.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            self.assertEqual(row["native_popc"], row["list_rows"] * row["words"] * 4 * row["queries"])
            self.assertLessEqual(row["shared_popc"], row["identical_popc"])
            self.assertLessEqual(row["shared_popc"], row["complement_popc"])
        for ds in result["datasets"]:
            for summary in ds["summaries"]:
                rs = [r for r in rows if r["dataset"] == ds["dataset"] and r["scale_multiplier"] == summary["scale_multiplier"] and r["batch"] == summary["batch"]]
                self.assertEqual(len(rs), summary["group_count"])
                for key in ("native_popc", "local_popc", "identical_popc", "complement_popc", "shared_popc", "weighted_updates", "descriptor_bytes"):
                    self.assertEqual(sum(r[key] for r in rs), summary[key])


if __name__ == "__main__":
    unittest.main()
