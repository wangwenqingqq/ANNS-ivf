"""Prepare bounded public-benchmark prefixes; never copy private source paths."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, dim, rows, offset=0):
    with Path(path).open("rb") as stream:
        stream.seek(offset * (dim + 1) * 4)
        raw = stream.read(rows * (dim + 1) * 4)
    assert len(raw) == rows * (dim + 1) * 4
    records = np.frombuffer(raw, dtype="<i4").reshape(rows, dim + 1)
    assert np.all(records[:, 0] == dim), "fvecs dimension mismatch"
    values = records[:, 1:].copy().view("<f4")
    assert np.isfinite(values).all()
    return values, {"offset_rows": offset, "rows": rows, "dimension": dim,
                    "raw_fvecs_sha256": sha(raw), "array_sha256": sha(values.tobytes())}


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "bounded prefixes, not full-million datasets", "datasets": []}
    for name, dim, size, offset, base, query in (
            ("SIFT", 128, 32768, 262144, args.sift_base, args.sift_query),
            ("GIST", 960, 16384, 16384, args.gist_base, args.gist_query)):
        x, xb = read(base, dim, size)
        train, tb = read(base, dim, 8192, offset)
        q, qb = read(query, dim, 96)
        np.savez_compressed(args.out / (name + ".npz"), x=x, train=train, q=q)
        manifest["datasets"].append({"name": name, "base": xb, "training": tb,
                                     "queries": qb, "public_prefix_equivalence": "pending verification"})
    rng = np.random.default_rng(20261015)
    values = []
    for size in (8192, 8192, 96):
        array = rng.standard_normal((size, 960)).astype(np.float32)
        array /= np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
        values.append(array)
    np.savez_compressed(args.out / "Gaussian.npz", x=values[0], train=values[1], q=values[2])
    manifest["datasets"].append({"name": "Gaussian", "synthetic": True,
        "generator": "PCG64 seed 20261015; FP32 independent normal then FP64 norm / FP32 normalization",
        "arrays": {key: {"shape": list(v.shape), "sha256": sha(v.tobytes())}
                   for key, v in zip(("x", "train", "q"), values)}})
    (args.out / "INPUTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("sift-base", "sift-query", "gist-base", "gist-query"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
