"""Bounded public-prefix downloader; never download a complete multi-GB base."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

SIFT = "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/0bc1ed67fc2626bc6f110c8794a70572681b7395/sift1M/"
GIST = "https://huggingface.co/datasets/maknee/gist1m/resolve/a82497d163022c768ca9288b1d83f9cd0c97640b/fbin/"


def span(url, start, length):
    request = urllib.request.Request(url + "?prefix_start=" + str(start) + "&prefix_bytes=" + str(length),
                                    headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        cr = response.headers.get("Content-Range")
        if cr:
            assert cr.startswith("bytes " + str(start) + "-"), "wrong HTTP range"
        else:
            # Some mirrors ignore Range. Skip only a bounded requested offset.
            left = start
            while left:
                block = response.read(min(left, 1024 * 1024))
                assert block, "short HTTP skip"
                left -= len(block)
        chunks, left = [], length
        while left:
            block = response.read(min(left, 1024 * 1024))
            assert block, "short HTTP prefix"
            chunks.append(block)
            left -= len(block)
    return b"".join(chunks)


def fvec(raw, rows, dim):
    data = np.frombuffer(raw, dtype="<i4").reshape(rows, dim + 1)
    assert np.all(data[:, 0] == dim)
    return data[:, 1:].copy().view("<f4")


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    x = fvec(span(SIFT + "sift_base.fvecs", 0, 32768 * 516), 32768, 128)
    train = fvec(span(SIFT + "sift_base.fvecs", 262144 * 516, 8192 * 516), 8192, 128)
    q = fvec(span(SIFT + "sift_query.fvecs", 0, 96 * 516), 96, 128)
    np.savez_compressed(out / "SIFT.npz", x=x, train=train, q=q)
    raw = span(GIST + "base.fbin", 0, 8 + 24576 * 960 * 4)
    assert np.frombuffer(raw[:8], dtype="<u4").tolist() == [1000000, 960]
    all_x = np.frombuffer(raw[8:], dtype="<f4").reshape(24576, 960).copy()
    rawq = span(GIST + "queries.fbin", 0, 8 + 96 * 960 * 4)
    assert np.frombuffer(rawq[:8], dtype="<u4").tolist() == [1000, 960]
    q = np.frombuffer(rawq[8:], dtype="<f4").reshape(96, 960).copy()
    np.savez_compressed(out / "GIST.npz", x=all_x[:16384], train=all_x[16384:], q=q)
    result = {"SIFT_public_mirror": SIFT, "GIST_public_mirror": GIST,
              "official_original_archive_equivalence_verified": False, "datasets": {}}
    for name in ("SIFT", "GIST"):
        loaded = np.load(out / (name + ".npz"))
        result["datasets"][name] = {key: hashlib.sha256(loaded[key].tobytes()).hexdigest()
                                    for key in ("x", "train", "q")}
    (out / "PUBLIC_PREFIXES.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
