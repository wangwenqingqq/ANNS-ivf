#!/usr/bin/env python3
"""CPU exact-integer work probe; never a GPU benchmark or native index reproduction."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import platform
import numpy as np
import faiss
faiss.omp_set_num_threads(1)
HERE = Path(__file__).resolve().parent
FULL = (1 << 32) - 1
WEIGHTS = (1, 2, 4, -8)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def distance(x, q):
    return np.square(x.astype(np.float64) - q.astype(np.float64)).sum(1)


def nearest(x, centers):
    return np.column_stack([distance(x, c) for c in centers]).argmin(1)


def packed(codes):
    assert codes.dtype.kind == "i" and np.all((codes >= -8) & (codes <= 7))
    assert codes.shape[1] % 32 == 0
    unsigned = codes.astype(np.uint8) & 15
    places = (np.uint32(1) << np.arange(31, -1, -1, dtype=np.uint32))
    return np.stack([(((unsigned >> b) & 1).reshape(len(codes), -1, 32) * places).sum(2).astype(np.uint32) for b in range(4)], axis=1)


def dictionary(masks, complement):
    r = len(masks); maps = []; bases = []
    for row in masks:
        terms = {}; base = 0
        for raw, weight in zip(row, WEIGHTS):
            mask = int(raw)
            flipped = complement and (FULL ^ mask) < mask
            key = (FULL ^ mask) if flipped else mask
            if flipped:
                base += weight; weight = -weight
            if key:
                terms[key] = terms.get(key, 0) + weight
        maps.append({k: v for k, v in terms.items() if v}); bases.append(base)
    keys = sorted(set().union(*(x.keys() for x in maps)))
    coeff = np.array([[x.get(k, 0) for k in keys] for x in maps], dtype=np.int32).reshape(r, len(keys))
    base = np.array(bases, dtype=np.int32)
    return keys, coeff, base


def apply_plan(words, plan):
    keys, coeff, base = plan
    pc = np.array([[(int(x) & k).bit_count() for k in keys] for x in words], dtype=np.int32).reshape(len(words), len(keys))
    whole = np.array([int(x).bit_count() for x in words], dtype=np.int32)
    return pc @ coeff.T + whole[:, None] * base[None, :]


def native_word(words, masks):
    return np.array([[sum(w * (int(x) & int(m)).bit_count() for w, m in zip(WEIGHTS, row)) for row in masks] for x in words], dtype=np.int32)


def self_test():
    rng = np.random.default_rng(7); cases = 0
    for r in (1, 8, 32):
        for trial in range(12):
            codes = rng.integers(-8, 8, (r, 64), dtype=np.int8)
            if trial == 0: codes[:] = -1
            if trial == 1: codes[:] = 0
            if trial == 2: codes[:] = -8
            masks = packed(codes)
            words = np.r_[np.array([0, FULL], np.uint32), rng.integers(0, 1 << 32, 50, dtype=np.uint32)]
            bits = ((words[:, None] >> np.arange(31, -1, -1)) & 1).astype(np.int32)
            for word in range(2):
                direct = bits @ codes[:, word * 32:(word + 1) * 32].astype(np.int32).T
                original = native_word(words, masks[:, :, word])
                assert np.array_equal(original, direct)
                for complement in (False, True):
                    assert np.array_equal(apply_plan(words, dictionary(masks[:, :, word], complement)), direct)
                    cases += 1
    try:
        packed(np.array([[8] * 32], dtype=np.int8))
    except AssertionError: pass
    else: raise AssertionError("invalid code escaped")
    words = np.array([FULL], dtype=np.uint32)
    plan = dictionary(np.array([[1, 2, 4, 8]], dtype=np.uint32), True)
    damaged = (plan[0], plan[1].copy(), plan[2]); damaged[1][0, 0] += 1
    assert not np.array_equal(apply_plan(words, plan), apply_plan(words, damaged))
    return {"exact_integer_cases": cases, "invalid_code_rejected": True, "coefficient_mutation_detected": True}


def rescale(sample):
    a = np.abs(sample.astype(np.float64)); a /= np.linalg.norm(a)
    end = 17 / a.max(); start = end * .52
    cur = (start * a + 1e-5).astype(np.int64)
    denominator = len(a) * .25 + np.sum(cur * cur + cur)
    numerator = float(np.sum((cur + .5) * a)); best = -1.; result = start
    heap = [((int(u) + 1) / v, i) for i, (u, v) in enumerate(zip(cur, a))]
    heapq.heapify(heap)
    while heap:
        t, i = heapq.heappop(heap); cur[i] += 1
        denominator += 2 * cur[i]; numerator += a[i]
        cosine = numerator / np.sqrt(denominator)
        if cosine > best: best, result = cosine, t
        if cur[i] < 7 and (cur[i] + 1) / a[i] < end:
            heapq.heappush(heap, ((cur[i] + 1) / a[i], i))
    return result


def query_codes(rotated, scale):
    norms = np.sqrt(np.square(rotated.astype(np.float64)).sum(1)).astype(np.float32)
    unit = rotated / np.maximum(norms[:, None], np.float32(1e-30))
    return np.clip(np.rint(unit * np.float32(scale)), -8, 7).astype(np.int8)


def summarize(rows):
    totals = {k: sum(r[k] for r in rows) for k in ("native_popc", "local_popc", "identical_popc", "complement_popc", "shared_popc", "weighted_updates", "descriptor_bytes", "dictionary_key_input_bytes", "pair_metadata_read_write_bytes")}
    totals.update({"group_count": len(rows), "candidate_word_load_ratio_ideal": sum(r["list_rows"] * r["words"] for r in rows) / sum(r["list_rows"] * r["words"] * r["queries"] for r in rows),
                   "shared_to_native_popc": totals["shared_popc"] / totals["native_popc"],
                   "shared_to_local_popc": totals["shared_popc"] / totals["local_popc"],
                   "weighted_updates_to_native": totals["weighted_updates"] / totals["native_popc"],
                   "max_dictionary_scratch_model_bytes": max(r["scratch_bytes"] for r in rows)})
    return totals


def run_dataset(name, data, centers, labels, rotation, nprobe, out):
    x, train, q = (np.ascontiguousarray(data[k], dtype=np.float32) for k in ("x", "train", "q"))
    n, d = x.shape; assert q.shape == (96, d) and d % 32 == 0
    assert all(np.isfinite(v).all() for v in (x, train, q))
    lists = [np.flatnonzero(labels == c) for c in range(64)]
    ranks = np.array([np.lexsort((np.arange(64), distance(centers, v))) for v in q])
    recalls = []
    for qi, v in enumerate(q):
        ds = distance(x, v); gt = np.lexsort((np.arange(n), ds))[:10]
        recalls.append([np.isin(labels[gt], ranks[qi, :p]).mean() for p in (1, 4, 16, 64)])
    if nprobe is None:
        nprobe = next(p for p, score in zip((1, 4, 16, 64), np.mean(recalls[:32], axis=0)) if score >= .95)
    probes = ranks[:, :nprobe]
    rotated = np.ascontiguousarray(rotation.apply_py(q))
    rng = np.random.default_rng(7)
    scaling = float(np.float32(np.mean([rescale(v) for v in rng.normal(size=(100, d))])))
    validation_cases = 0; all_rows = []
    for mult in (.9, 1., 1.1):
        codes = query_codes(rotated, scaling * mult); masks = packed(codes)
        for batch in (1, 8, 32):
            for first in range(32, 96, batch):
                incidence = probes[first:first + batch]
                for c in np.unique(incidence):
                    qs = np.flatnonzero((incidence == c).any(1)) + first
                    r = len(qs); wc = d // 32; count = len(lists[c])
                    costs = {"local_popc": 0, "identical_popc": 0, "complement_popc": 0, "shared_popc": 0, "weighted_updates": 0, "descriptor_bytes": 0}
                    scratch = 0
                    validate = mult == 1. and batch == 32
                    if validate:
                        sample = x[lists[c][:32]] - centers[c]
                        sign = (rotation.apply_py(np.ascontiguousarray(sample)) >= 0).astype(np.int8)
                        places = np.uint32(1) << np.arange(31, -1, -1, dtype=np.uint32)
                        words = (sign.reshape(len(sign), wc, 32) * places).sum(2).astype(np.uint32)
                        recovered = np.zeros((len(sign), r), dtype=np.int32)
                    for w in range(wc):
                        planes = masks[qs, :, w]
                        simple = dictionary(planes, False); shared = dictionary(planes, True)
                        complement_slots = len(shared[0]) + int(np.any(shared[2]))
                        if len(simple[0]) <= complement_slots:
                            shared = simple
                        keys, coeff, base = shared
                        slots = len(keys) + int(np.any(base))
                        nnz = int(np.count_nonzero(coeff)); bnnz = int(np.count_nonzero(base))
                        local = 0
                        for row in planes:
                            one = dictionary(row[None], False)
                            two = dictionary(row[None], True)
                            local += min(len(one[0]), len(two[0]) + int(np.any(two[2])))
                        costs["local_popc"] += count * local
                        costs["identical_popc"] += count * len(simple[0])
                        costs["complement_popc"] += count * complement_slots
                        costs["shared_popc"] += count * slots
                        costs["weighted_updates"] += count * (nnz + bnnz)
                        costs["descriptor_bytes"] += 4 * len(keys) + 2 * nnz + 2 * (r + 1) + r + 8
                        scratch = max(scratch, 32 * slots + 32 * r * 4)
                        if validate:
                            ref = native_word(words[:, w], planes)
                            actual = apply_plan(words[:, w], shared)
                            assert np.array_equal(actual, ref), "factoring mismatch"
                            recovered += actual; validation_cases += actual.size
                    if validate:
                        assert np.array_equal(recovered, sign.astype(np.int32) @ codes[qs].astype(np.int32).T), "dense parity mismatch"
                    all_rows.append({"dataset": name, "scale_multiplier": mult, "batch": batch, "first_query": first, "cluster": int(c), "queries": r, "list_rows": count, "words": wc,
                                     "native_popc": count * wc * 4 * r, **costs, "scratch_bytes": scratch,
                                     "dictionary_key_input_bytes": r * wc * 4 * 4,
                                     "pair_metadata_read_write_bytes": r * 8 * 2})
    summary = [{"scale_multiplier": mult, "batch": b, **summarize([r for r in all_rows if r["scale_multiplier"] == mult and r["batch"] == b])} for mult in (.9, 1., 1.1) for b in (1, 8, 32)]
    result = {"dataset": name, "n": n, "d": d, "nprobe": nprobe, "calibration_routing_recalls": np.mean(recalls[:32], axis=0).tolist(),
              "evaluation_routing_recall": float(np.mean(np.asarray(recalls[32:])[:, (1, 4, 16, 64).index(nprobe)])),
              "input_hashes": {k: sha(np.ascontiguousarray(data[k]).tobytes()) for k in ("x", "train", "q")},
              "centers_sha256": sha(centers.tobytes()), "labels_sha256": sha(labels.tobytes()),
              "cpu_query_scaling": scaling, "integer_output_comparisons": validation_cases, "summaries": summary}
    print("PASS", name, n, d, nprobe, validation_cases, flush=True)
    return result, all_rows


def cpu_model():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"experiment": "ivf_20261017_query_mask_factorization_cpu_v1", "scope": "exact integer primitive and work-count opportunity only; no native GPU output, traffic or speed claim",
              "software": {"python": platform.python_version(), "numpy": np.__version__, "faiss_cpu": faiss.__version__, "cpu": cpu_model(), "arch": platform.machine()},
              "source_sha256": sha(Path(__file__).read_bytes()), "mechanism_preregistered_sha256": sha((HERE / "MECHANISM.md").read_bytes()),
              "validation": self_test(), "datasets": []}
    all_rows = []
    for name in ("SIFT", "GIST", "Gaussian"):
        data = dict(np.load(args.previous / "inputs/primary" / (name + ".npz")))
        models = args.previous / "runs/primary_01/out/models" / name
        centers, labels = np.load(models / "centers.npy"), np.load(models / "labels.npy")
        wrapper = faiss.read_index(str(models / "rabitq_1.faiss"))
        rotation = faiss.downcast_VectorTransform(wrapper.chain.at(0))
        obj, rows = run_dataset(name, data, centers, labels, rotation, 64 if name == "Gaussian" else 16, args.out)
        result["datasets"].append(obj); all_rows.extend(rows)
        if name == "SIFT" and args.large_sift:
            raw = np.fromfile(args.large_sift, dtype="<i4", count=262144 * 129).reshape(262144, 129)
            assert np.all(raw[:, 0] == 128)
            large = {**data, "x": raw[:, 1:].copy().view("<f4")}
            expanded = nearest(large["x"], centers).astype(np.int64)
            assert np.array_equal(expanded[:len(labels)], labels)
            obj, rows = run_dataset("SIFT262k", large, centers, expanded, rotation, 16, args.out)
            result["datasets"].append(obj); all_rows.extend(rows)
    if args.glove:
        data = dict(np.load(args.glove)); train = data["train"]; d = train.shape[1]
        clus = faiss.Clustering(d, 64); clus.niter = 12; clus.nredo = 1; clus.seed = 20261017
        clus.train(train, faiss.IndexFlatL2(d))
        centers = faiss.vector_to_array(clus.centroids).reshape(64, d).copy()
        labels = nearest(data["x"], centers).astype(np.int64)
        rotation = faiss.RandomRotationMatrix(d, d); rotation.init(123)
        obj, rows = run_dataset("GloVe100", data, centers, labels, rotation, None, args.out)
        result["datasets"].append(obj); all_rows.extend(rows)
    (args.out / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.out / "GROUPS.jsonl").open("w") as f:
        for r in all_rows: f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    print("PASS: exact integer work screen, no GPU access", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--large-sift", type=Path)
    parser.add_argument("--glove", type=Path)
    args = parser.parse_args()
    if args.out:
        assert args.previous is not None; run(args)
    else: print(self_test())
