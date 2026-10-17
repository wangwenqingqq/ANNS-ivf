#!/usr/bin/env python3
"""Extract bounded public GloVe arrays; archive and derived arrays remain off Git."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    assert not args.out.exists() and not args.out.with_suffix(".json").exists(), "refuse overwrite"
    with h5py.File(args.source, "r") as f:
        print({k: list(f[k].shape) for k in f})
        x = f["train"][:40960].astype(np.float32)
        q = f["test"][:96].astype(np.float32)
    arrays = {}
    for k, a in (("x", x[:32768]), ("train", x[32768:]), ("q", q)):
        norms = np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
        assert np.isfinite(a).all() and (norms > 0).all()
        a = (a.astype(np.float64) / norms).astype(np.float32)
        arrays[k] = np.pad(a, ((0, 0), (0, 28)))
    np.savez_compressed(args.out, **arrays)
    h = hashlib.sha256()
    with args.source.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    manifest = {"url": "https://ann-benchmarks.com/glove-100-angular.hdf5",
                "archive_bytes": args.source.stat().st_size, "archive_sha256": h.hexdigest(),
                "normalization": "FP64 L2 normalization cast FP32; 100D padded to128",
                "arrays": {k: {"shape": list(a.shape), "sha256": sha(a.tobytes())} for k, a in arrays.items()}}
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
