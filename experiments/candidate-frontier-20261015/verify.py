#!/usr/bin/env python3
"""Independent receipt replay; exhaustive distances are verification-only."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import numpy as np


def sha(value):
    return hashlib.sha256(value).hexdigest()


def distance(x, q):
    return np.square(x.astype(np.float64) - q.astype(np.float64)).sum(axis=1)


def best(ids, ds, count=10):
    return ids[np.lexsort((ids, ds))[:count]]


def certificate(name, x, train):
    d = x.shape[1]
    maxnorm = np.square(x.astype(np.float64)).sum(1).max()
    if name.startswith("norm"):
        groups = int(name.split("_")[1])
        norms = np.linalg.norm(x.astype(np.float64).reshape(len(x), groups, -1), axis=2)
    if name == "sq8":
        lo = train.astype(np.float64).min(0)
        step = (train.astype(np.float64).max(0) - lo) / 255
        step[step == 0] = 1
        codes = np.rint(np.clip((x.astype(np.float64) - lo) / step, 0, 255)).astype(np.uint8)
        decoded = lo + codes.astype(np.float64) * step
        residual = np.nextafter(np.linalg.norm(x.astype(np.float64) - decoded, axis=1), np.inf)
    def lower(q):
        guard = 64 * np.finfo(np.float64).eps * d * (maxnorm + np.square(q.astype(np.float64)).sum() + 1)
        if name.startswith("prefix"):
            m = d // int(name.split("_")[1]); lb = distance(x[:, :m], q[:m])
        elif name.startswith("norm"):
            qnorm = np.linalg.norm(q.astype(np.float64).reshape(groups, -1), axis=1)
            lb = np.square(norms - qnorm).sum(1)
        else:
            lb = np.square(np.maximum(0, np.sqrt(distance(decoded, q)) - residual))
        return np.maximum(0, lb - guard)
    return lower


def replay(args):
    results = json.loads((args.evidence / "RESULTS.json").read_text())
    with gzip.open(args.evidence / "QUERY.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    lookup = {(r["dataset"], r["query"], r["nprobe"], r["method"], r.get("refine_pool", -1)): r for r in rows}
    assert len(lookup) == len(rows)
    count = {"queries": 0, "routing_cases": 0, "certificate_timelines": 0, "native_rows": 0}
    for meta in results["datasets"]:
        name, n, d = meta["name"], meta["n"], meta["d"]
        if name not in args.datasets:
            continue
        data = np.load(args.inputs / (name + ".npz"))
        x, train, queries = [data[k] for k in ("x", "train", "q")]
        for k in ("x", "train", "q"):
            assert sha(data[k].tobytes()) == meta["input_hashes"][k]
        centers = np.load(args.models / name / "centers.npy")
        labels = np.load(args.models / name / "labels.npy")
        assert sha(centers.tobytes()) == meta["centers_sha256"]
        assert sha(labels.tobytes()) == meta["assignments_sha256"]
        independent_labels = np.column_stack([distance(x, c) for c in centers]).argmin(1)
        assert np.array_equal(independent_labels, labels)
        lists = [np.flatnonzero(labels == i) for i in range(64)]
        methods = {m: certificate(m, x, train) for m in ("prefix_8", "prefix_2", "norm_8", "norm_16", "sq8")}
        native = {}
        if args.native:
            import faiss
            faiss.omp_set_num_threads(1)
            for kind in ("flat", "pq8", "rabitq_1", "rabitq_4"):
                path = args.models / name / (kind + ".faiss")
                assert sha(path.read_bytes()) == meta["storage"][kind]["serialized_sha256"]
                wrapper = faiss.read_index(str(path))
                index = faiss.downcast_index(wrapper.index) if kind.startswith("rabitq") else wrapper
                rotation = faiss.downcast_VectorTransform(wrapper.chain.at(0)) if kind.startswith("rabitq") else None
                for c in range(64):
                    actual = faiss.rev_swig_ptr(index.invlists.get_ids(c), index.invlists.list_size(c)).copy()
                    assert np.array_equal(np.sort(actual), lists[c])
                native[kind] = (wrapper, index, rotation)
        for qi in range(32, 96):
            q = queries[qi]; exact = distance(x, q); gt = best(np.arange(n), exact)
            ranked = np.lexsort((np.arange(64), distance(centers, q)))
            bounds = {m: fn(q) for m, fn in methods.items()}
            for lb in bounds.values():
                assert np.all(lb <= exact + 1e-9 * (exact.max() + 1))
            count["queries"] += 1
            for p in (1, 4, 16, 64):
                probes = ranked[:p].astype(np.int64)
                ids = np.concatenate([lists[i] for i in probes]); c = len(ids)
                answer = best(ids, exact[ids]); tau_final = exact[answer[-1]]
                row = lookup[name, qi, p, "perfect_answer_oracle", -1]
                assert row["global_gt"] == gt.tolist() and row["candidate_exact_ids"] == answer.tolist()
                assert row["read_bytes"] == 10 * d * 4 + c * 8
                count["routing_cases"] += 1
                for m, lbs in bounds.items():
                    b = lbs[ids]
                    for mode in ("final_tau_oracle", "online"):
                        seed = 0 if mode == "final_tau_oracle" else min(64, c)
                        if seed == 0:
                            mask = b <= tau_final
                        else:
                            mask = np.zeros(c, bool); mask[:seed] = True
                            selected = ids[:seed]; threshold = exact[best(selected, exact[selected])[-1]]
                            for start in range(seed, c, 64):
                                # Already-read distances update the replay threshold only after admission.
                                keep = b[start:start + 64] <= threshold
                                mask[start:start + len(keep)] = keep
                                selected = np.concatenate((selected, ids[start:start + 64][keep]))
                                threshold = exact[best(selected, exact[selected])[-1]]
                        r = lookup[name, qi, p, m + ":" + mode, -1]
                        assert best(ids[mask], exact[ids[mask]]).tolist() == r["returned_ids"] == answer.tolist()
                        full = int(mask.sum()); e = meta["storage"][m]["evidence_input_bytes_per_candidate"]
                        evidence = (c - seed) * e; reads = evidence + full * 4 * d + c * 8
                        assert r["full_rows"] == full and r["evidence_read_bytes"] == evidence
                        assert r["read_bytes_refetch"] == reads
                        reuse = reads - (full - seed) * e if m.startswith("prefix") else reads
                        assert r["read_bytes_prefix_reuse_optimistic"] == reuse
                        assert r["survival_mask_sha256"] == sha(np.packbits(mask).tobytes())
                        assert r["modeled_read_write_bytes"] == reads + c + (full - seed) * 8
                        for g in (1, 8, 32):
                            expanded = sum(min(g, c - s) for s in range(0, c, g) if mask[s:s + g].any())
                            assert r["grain_models"][str(g)] == {"full_rows": expanded, "read_bytes_refetch": evidence + expanded * d * 4 + c * 8}
                        count["certificate_timelines"] += 1
                for kind, (_, index, rotation) in native.items():
                    query = np.ascontiguousarray(rotation.apply_py(q[None]) if rotation else q[None])
                    ct = index.quantizer.reconstruct_n(0, 64)
                    coarse = distance(ct[probes], query[0]).astype(np.float32)[None]
                    index.nprobe = p; faiss.cvar.indexIVF_stats.reset()
                    _, pool = index.search_preassigned(query, min(128, c), probes[None], coarse)
                    for refine in ((0,) if kind == "flat" else (0, 10, 32, 128)):
                        selected = pool[0, :10] if refine == 0 else best(pool[0, :refine], exact[pool[0, :refine]])
                        r = lookup[name, qi, p, kind, refine]
                        assert selected.tolist() == r["returned_ids"]
                        assert r["native_distance_counter"] == faiss.cvar.indexIVF_stats.ndis
                        assert r["read_bytes"] == c * (index.code_size + 8) + min(refine, len(pool[0])) * d * 4
                        count["native_rows"] += 1
            print("VERIFIED", name, qi + 1, flush=True) if qi % 16 == 15 else None
    report = {"scope": "independent direct-sum receipt replay; no GPU or timing claim",
              "datasets": args.datasets, "native_saved_model_replay": args.native,
              "python": platform.python_version(), "numpy": np.__version__, "arch": platform.machine(),
              "counts": count, "verifier_sha256": sha(Path(__file__).read_bytes()),
              "result_sha256": sha((args.evidence / "RESULTS.json").read_bytes()), "passed": True}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("inputs", "models", "evidence", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["SIFT", "GIST", "Gaussian"])
    parser.add_argument("--native", action="store_true")
    replay(parser.parse_args())
