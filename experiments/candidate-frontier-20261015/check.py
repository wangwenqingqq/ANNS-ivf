#!/usr/bin/env python3
"""CPU opportunity ledger, not a GPU performance experiment."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import faiss
from certificates import Certificate, bytes_ledger, distances, online, self_test, top
from native import Native

HERE = Path(__file__).resolve().parent
PROBES = (1, 4, 16, 64)
METHODS = ("prefix_8", "prefix_2", "norm_8", "norm_16", "sq8")
faiss.omp_set_num_threads(1)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + "\n")


def matrix(a, b):
    return np.column_stack([distances(a, row) for row in b])


def choose(ids, values, count=10):
    return top(ids, values, count)[0]


def recall(answer, gt):
    return len(set(answer.tolist()) & set(gt.tolist())) / 10


def summary(values):
    a = np.asarray(values)
    return {"count": len(a), "mean": float(a.mean()), "p50": float(np.quantile(a, .5)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def train_centers(train):
    clus = faiss.Clustering(train.shape[1], 64)
    clus.niter, clus.nredo, clus.seed = 12, 1, 20261015
    index = faiss.IndexFlatL2(train.shape[1])
    clus.train(np.ascontiguousarray(train), index)
    return faiss.vector_to_array(clus.centroids).reshape(64, train.shape[1]).copy()


def run_dataset(name, data, out):
    x, train, queries = [np.ascontiguousarray(data[k], dtype=np.float32) for k in ("x", "train", "q")]
    assert len(queries) == 96 and len(train) == 8192
    assert all(np.isfinite(v).all() for v in (x, train, queries))
    n, d = x.shape
    print("DATASET", name, n, d, flush=True)
    model_dir = out / "models" / name
    model_dir.mkdir(parents=True)
    centers = train_centers(train)
    labels = matrix(x, centers).argmin(1).astype(np.int64)
    lists = [np.flatnonzero(labels == c) for c in range(64)]
    np.save(model_dir / "centers.npy", centers)
    np.save(model_dir / "labels.npy", labels)
    native, storage, preparation = {}, {}, {}
    raw_bytes = x.nbytes
    for kind in ("flat", "pq8", "rabitq_1", "rabitq_4"):
        started = time.perf_counter()
        native[kind] = Native(kind, x, train, centers, labels)
        preparation[kind] = {"cpu_build_seconds_observation_only": time.perf_counter() - started,
                             "training_rows": len(train), "raw_encoding_input_bytes": raw_bytes}
        storage[kind] = native[kind].metadata(raw_bytes)
        faiss.write_index(native[kind].wrapper, str(model_dir / (kind + ".faiss")))
        print("NATIVE", name, kind, storage[kind]["code_bytes_per_vector"], flush=True)
    static_budget = int(storage["flat"]["serialized_index_bytes"] + .35 * raw_bytes)
    certs = [Certificate(kind, x, train) for kind in METHODS]
    for cert in certs:
        footprint = raw_bytes + len(x) * 8 + centers.nbytes + cert.extra_bytes + cert.shared_bytes
        storage[cert.name] = {"raw_payload_bytes": raw_bytes, "extra_evidence_bytes": cert.extra_bytes,
                              "shared_evidence_bytes": cert.shared_bytes,
                              "declared_static_storage_bytes": footprint,
                              "static_budget_admissible": footprint <= static_budget,
                              "offline_encoding_raw_input_bytes": raw_bytes,
                              "evidence_input_bytes_per_candidate": cert.evidence_bytes,
                              "query_evidence_preparation_input_bytes": d * 4,
                              "candidate_arithmetic_dimension": d if cert.name == "sq8" else
                                  cert.m if cert.name.startswith("prefix") else cert.m}
    all_rows, quality, batch_masks, gaps = [], [], {}, []
    ids = np.arange(n, dtype=np.int64)
    for qi, q in enumerate(queries):
        ref = distances(x, q)
        # Independent subtract/square/sum check, not norm-dot cancellation.
        direct = np.square(x[:32].astype(np.float64) - q.astype(np.float64)).sum(1)
        assert np.allclose(ref[:32], direct, rtol=1e-13, atol=1e-12)
        global11 = choose(ids, ref, 11)
        gt = global11[:10]
        gap = float((ref[global11[10]] - ref[global11[9]]) / max(ref[global11[9]], 1e-300))
        gaps.append(gap)
        center_dist = distances(centers, q)
        ranked_lists = np.lexsort((np.arange(64), center_dist))
        for probe in PROBES:
            probes = ranked_lists[:probe].astype(np.int64)
            candidates = np.concatenate([lists[c] for c in probes])
            count = len(candidates)
            assert count >= 10 and len(np.unique(candidates)) == count
            exact_ids = choose(candidates, ref[candidates])
            tau = float(ref[exact_ids[-1]])
            candidate_recall = recall(exact_ids, gt)
            base = {"dataset": name, "query": qi, "split": "calibration" if qi < 32 else "evaluation",
                    "nprobe": probe, "candidates": count, "global_gt": gt.tolist(),
                    "candidate_exact_ids": exact_ids.tolist(), "ivf_global_recall": candidate_recall,
                    "global_gap": gap, "candidate_payload_scan_bytes": count * d * 4,
                    "baseline_candidate_read_bytes": count * (d * 4 + 8),
                    "common_query_input_bytes": d * 4, "common_cold_centroid_bytes": centers.nbytes,
                    "final_threshold_oracle": tau}
            quality.append(base)
            if qi < 32:
                continue
            all_rows.append({**base, "method": "perfect_answer_oracle", "hindsight": True,
                             "selector_cost_omitted": True, "read_bytes": 10 * d * 4 + count * 8,
                             "global_recall": candidate_recall, "returned_ids": exact_ids.tolist()})
            for cert in certs:
                lbs = np.concatenate([cert.bound(candidates[s:s + 1024], q) for s in range(0, count, 1024)])
                assert np.all(lbs <= ref[candidates] + cert.guard(q)), "invalid certificate"
                final_keep = lbs <= tau
                assert set(exact_ids.tolist()) <= set(candidates[final_keep].tolist())
                oracle = bytes_ledger(cert, count, final_keep, 0)
                all_rows.append({**base, **oracle, "method": cert.name + ":final_tau_oracle",
                                 "hindsight": True, "global_recall": candidate_recall,
                                 "returned_ids": exact_ids.tolist()})
                answer, mask, seed = online(cert, candidates, q)
                assert np.array_equal(answer, exact_ids), "online selected-list exactness mismatch"
                all_rows.append({**base, **bytes_ledger(cert, count, mask, seed),
                                 "method": cert.name + ":online", "hindsight": False,
                                 "global_recall": recall(answer, gt), "returned_ids": answer.tolist()})
                batch_masks[probe, cert.name, qi] = candidates[mask]
            for kind, index in native.items():
                pool, observed_ndis = index.search(q, probes, count)
                assert set(pool.tolist()) <= set(candidates.tolist()), "native search escaped common lists"
                extra = {"method": kind, "native_distance_counter": observed_ndis,
                         "code_read_bytes_model": count * index.index.code_size,
                         "source_id_read_bytes_model": count * 8,
                         "serialized_index_bytes": index.serialized_bytes}
                for refine in ((0,) if kind == "flat" else (0, 10, 32, 128)):
                    selected = pool[:10] if refine == 0 else choose(pool[:refine], distances(x[pool[:refine]], q))
                    raw_rows = 0 if refine == 0 else min(refine, len(pool))
                    footprint = index.serialized_bytes + (raw_bytes if raw_rows else 0)
                    all_rows.append({**base, **extra, "refine_pool": refine, "refined_raw_rows": raw_rows,
                                     "global_recall": recall(selected, gt), "returned_ids": selected.tolist(),
                                     "raw_refinement_read_bytes": raw_rows * d * 4,
                                     "full_rows": count if kind == "flat" else raw_rows,
                                     "read_bytes": count * (index.index.code_size + 8) + raw_rows * d * 4,
                                     "static_storage_bytes": footprint,
                                     "static_budget_admissible": footprint <= static_budget,
                                     "cpu_control_not_gpu_performance": True})
        if qi % 16 == 15:
            print("QUERY", name, qi + 1, "/ 96", flush=True)
    calibration = {}
    for target in (.90, .95):
        means = {probe: float(np.mean([r["ivf_global_recall"] for r in quality if r["query"] < 32 and r["nprobe"] == probe])) for probe in PROBES}
        chosen = next((p for p in PROBES if means[p] >= target), None)
        calibration[str(target)] = {"chosen_nprobe": chosen, "calibration_recalls": means,
            "evaluation_recall": float(np.mean([r["ivf_global_recall"] for r in quality if r["query"] >= 32 and r["nprobe"] == chosen])) if chosen else None}
    cutoffs = np.quantile(np.asarray(gaps[32:]), [.25, .5, .75])
    for row in all_rows:
        row["gap_quartile_0_hardest"] = int(np.searchsorted(cutoffs, row["global_gap"], side="right"))
    batches = []
    for probe in PROBES:
        for method in METHODS:
            for size in (1, 8, 32):
                for first in range(32, 96, size):
                    masks = [batch_masks[probe, method, qi] for qi in range(first, first + size)]
                    counts = [len(v) for v in masks]
                    batches.append({"nprobe": probe, "method": method, "batch_size": size, "first_query": first,
                                    "sum_refined_rows": sum(counts), "max_refined_rows": max(counts),
                                    "mean_refined_rows": float(np.mean(counts)),
                                    "union_refined_rows": len(np.unique(np.concatenate(masks))),
                                    "scope": "cache/reuse and padding opportunity only; not GPU traffic or time"})
    aggregates = []
    keys = sorted({(r["nprobe"], r["method"], r.get("refine_pool", -1)) for r in all_rows})
    for probe, method, refine in keys:
        rows = [r for r in all_rows if r["nprobe"] == probe and r["method"] == method and r.get("refine_pool", -1) == refine]
        aggregates.append({"nprobe": probe, "method": method, "refine_pool": refine,
            "global_recall": summary([r["global_recall"] for r in rows]),
            "candidate_stage_read_ratio": summary([r.get("read_bytes_refetch", r.get("read_bytes")) / r["baseline_candidate_read_bytes"] for r in rows]),
            "mean_full_rows": float(np.mean([r.get("full_rows", r.get("refined_raw_rows", 10)) for r in rows]))})
    return {"name": name, "n": n, "d": d, "synthetic": name == "Gaussian",
            "input_hashes": {key: digest(data[key].tobytes()) for key in ("x", "train", "q")},
            "centers_sha256": digest(centers.tobytes()), "assignments_sha256": digest(labels.tobytes()),
            "cluster_sizes": [len(v) for v in lists], "storage": storage, "static_budget_bytes": static_budget,
            "cpu_preparation_observations": preparation, "calibration": calibration,
            "aggregates": aggregates, "batch_models": batches}, all_rows, quality


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    test = self_test()
    result = {"experiment": "ivf_20261015_candidate_information_cpu_v1",
              "scope": "CPU opportunity/quality and modeled bytes only; GPU performance remains pending",
              "software": {"python": platform.python_version(), "numpy": np.__version__, "faiss_cpu": faiss.__version__, "cpu_arch": platform.machine()},
              "source_hashes": {name: digest((HERE / name).read_bytes()) for name in
                                ("check.py", "certificates.py", "native.py")},
              "contract_sha256": digest((HERE / "CONTRACT.md").read_bytes()), "validation": test, "datasets": []}
    with (args.out / "QUERY.jsonl").open("w") as stream, (args.out / "ROUTING.jsonl").open("w") as routing:
        for name in args.datasets:
            data = np.load(args.inputs / (name + ".npz"))
            dataset, rows, quality = run_dataset(name, data, args.out)
            result["datasets"].append(dataset)
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            for row in quality:
                routing.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    json_write(args.out / "RESULTS.json", result)
    with (args.out / "QUERY.jsonl.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            compressed.write((args.out / "QUERY.jsonl").read_bytes())
    print("PASS: CPU opportunity screen; no GPU speed/physical-traffic/novelty claim", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs" / "primary")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["SIFT", "GIST", "Gaussian"])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.out:
        run(args)
    else:
        print(self_test())
