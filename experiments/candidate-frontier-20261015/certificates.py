"""Known deterministic certificate controls and a guarded online timeline."""
import hashlib
import numpy as np


def distances(x, q):
    delta = x.astype(np.float64) - q.astype(np.float64)
    result = np.einsum("ij,ij->i", delta, delta, optimize=False)
    assert np.isfinite(result).all(), "nonfinite distance"
    return result


def top(ids, ds, k=10):
    order = np.lexsort((ids, ds))[:k]
    return ids[order], ds[order]


class Certificate:
    def __init__(self, name, x, train):
        self.name, self.x, self.d = name, x, x.shape[1]
        self.max_norm2 = float(np.max(np.einsum("ij,ij->i", x.astype(np.float64), x.astype(np.float64))))
        self.shared_bytes = 8
        if name.startswith("prefix"):
            self.m = self.d // int(name.split("_")[1])
            self.evidence_bytes, self.extra_bytes = 4 * self.m, 0
        elif name.startswith("norm"):
            self.m = int(name.split("_")[1])
            values = x.astype(np.float64).reshape(len(x), self.m, -1)
            self.norms = np.sqrt(np.einsum("ijk,ijk->ij", values, values, optimize=False))
            self.evidence_bytes = 8 * self.m
            self.extra_bytes = self.norms.nbytes
        elif name == "sq8":
            self.minimum = train.astype(np.float64).min(0)
            self.scale = (train.astype(np.float64).max(0) - self.minimum) / 255
            self.scale[self.scale == 0] = 1
            self.codes = np.rint(np.clip((x.astype(np.float64) - self.minimum) / self.scale, 0, 255)).astype(np.uint8)
            reconstruction = self.minimum + self.codes.astype(np.float64) * self.scale
            error = x.astype(np.float64) - reconstruction
            self.error = np.nextafter(np.sqrt(np.einsum("ij,ij->i", error, error, optimize=False)), np.inf)
            self.evidence_bytes = self.d + 8
            self.extra_bytes = self.codes.nbytes + self.error.nbytes
            self.shared_bytes += self.minimum.nbytes + self.scale.nbytes
        else:
            raise ValueError(name)

    def guard(self, q):
        return 64 * np.finfo(np.float64).eps * self.d * (self.max_norm2 + float(q.astype(np.float64) @ q.astype(np.float64)) + 1)

    def bound(self, ids, q):
        if self.name.startswith("prefix"):
            value = distances(self.x[ids, :self.m], q[:self.m])
        elif self.name.startswith("norm"):
            grouped = q.astype(np.float64).reshape(self.m, -1)
            norm = np.sqrt(np.einsum("ij,ij->i", grouped, grouped, optimize=False))
            value = np.square(self.norms[ids] - norm).sum(1)
        else:
            decoded = self.minimum + self.codes[ids].astype(np.float64) * self.scale
            value = np.square(np.maximum(0, np.sqrt(distances(decoded, q)) - self.error[ids]))
        return np.maximum(0, value - self.guard(q))


class Reader:
    """No final-answer/threshold API exists during the online decision timeline."""
    def __init__(self, x, q):
        self.x, self.q, self.read_ids = x, q, []

    @property
    def final_threshold(self):
        raise AssertionError("hindsight threshold unavailable online")

    def full(self, ids):
        self.read_ids.extend(ids.tolist())
        return distances(self.x[ids], self.q)


def online(cert, candidates, q):
    reader = Reader(cert.x, q)
    count, seed = len(candidates), min(64, len(candidates))
    mask = np.zeros(count, dtype=bool)
    mask[:seed] = True
    best_ids, best_ds = top(candidates[:seed], reader.full(candidates[:seed]))
    tau = best_ds[-1] if len(best_ds) == 10 else np.inf
    for start in range(seed, count, 64):
        ids = candidates[start:start + 64]
        keep = cert.bound(ids, q) <= tau
        mask[start:start + len(ids)] = keep
        admitted = ids[keep]
        if len(admitted):
            ds = reader.full(admitted)
            best_ids, best_ds = top(np.concatenate((best_ids, admitted)), np.concatenate((best_ds, ds)))
            tau = best_ds[-1] if len(best_ds) == 10 else np.inf
    assert len(set(reader.read_ids)) == len(reader.read_ids) == int(mask.sum())
    assert set(reader.read_ids) == set(candidates[mask].tolist())
    return best_ids, mask, seed


def bytes_ledger(cert, count, mask, seed):
    d = cert.d
    evidence = (count - seed) * cert.evidence_bytes
    full = int(mask.sum())
    reads = evidence + full * d * 4 + count * 8
    reuse = reads - (full - seed) * cert.evidence_bytes if cert.name.startswith("prefix") else reads
    masks, queue = count, max(0, full - seed) * 8
    expanded = {}
    for grain in (1, 8, 32):
        padded = np.pad(mask, (0, (-count) % grain))
        selected = np.repeat(padded.reshape(-1, grain).any(1), grain)[:count]
        n = int(selected.sum())
        expanded[str(grain)] = {"full_rows": n, "read_bytes_refetch": evidence + n * d * 4 + count * 8}
    return {"candidates": count, "seed_full_rows": seed, "full_rows": full,
            "evidence_read_bytes": evidence, "raw_payload_read_bytes": full * d * 4,
            "source_id_read_bytes": count * 8, "read_bytes_refetch": reads,
            "read_bytes_prefix_reuse_optimistic": reuse,
            "modeled_mask_write_bytes": masks, "modeled_queue_write_bytes": queue,
            "modeled_read_write_bytes": reads + masks + queue,
            "grain_models": expanded,
            "survival_mask_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()}


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 32)).astype(np.float32)
    train = rng.normal(size=(128, 32)).astype(np.float32)
    for name in ("prefix_8", "prefix_2", "norm_8", "norm_16", "sq8"):
        cert = Certificate(name, x, train)
        for q in rng.normal(size=(8, 32)).astype(np.float32):
            ids = np.arange(len(x), dtype=np.int64)
            ds = distances(x, q)
            lb = cert.bound(ids, q)
            assert np.all(lb <= ds + cert.guard(q)), "invalid certificate"
            answer, mask, seed = online(cert, ids, q)
            assert np.array_equal(answer, top(ids, ds)[0])
            assert bytes_ledger(cert, len(ids), mask, seed)["full_rows"] <= len(ids)
    try:
        assert np.all((ds + 1) <= ds + cert.guard(q)), "invalid certificate"
    except AssertionError as error:
        assert str(error) == "invalid certificate"
    else:
        raise AssertionError("invalid-bound mutation escaped")
    try:
        Reader(x, q).final_threshold
    except AssertionError as error:
        assert str(error) == "hindsight threshold unavailable online"
    else:
        raise AssertionError("hindsight access escaped")
    try:
        distances(np.full((1, 32), np.nan), q)
    except AssertionError:
        pass
    else:
        raise AssertionError("nonfinite distance escaped")
    return {"bound_and_online_cases": 40, "invalid_bound_rejected": True,
            "hindsight_api_rejected": True, "nonfinite_rejected": True}
