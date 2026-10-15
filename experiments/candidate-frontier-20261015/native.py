"""Pinned native CPU Faiss controls with externally frozen list assignments."""
import hashlib
import numpy as np
import faiss
from certificates import distances


class Native:
    def __init__(self, kind, x, train, centers, labels):
        self.kind, self.d = kind, x.shape[1]
        self.rotation = None
        if kind.startswith("rabitq"):
            self.rotation = faiss.RandomRotationMatrix(self.d, self.d)
            self.rotation.init(123)
        xt, tt, ct = [self.transform(v) for v in (x, train, centers)]
        self.centers = ct
        self.quantizer = faiss.IndexFlatL2(self.d)
        self.quantizer.add(ct)
        if kind == "flat":
            self.index = faiss.IndexIVFFlat(self.quantizer, self.d, len(ct), faiss.METRIC_L2)
        elif kind == "pq8":
            self.index = faiss.IndexIVFPQ(self.quantizer, self.d, len(ct), 16, 8)
            self.index.pq.cp.niter = 12
            self.index.pq.cp.seed = 20261015
        else:
            bits = int(kind.split("_")[1])
            self.index = faiss.IndexIVFRaBitQ(self.quantizer, self.d, len(ct), faiss.METRIC_L2, True, bits)
            self.index.qb = 4
        self.index.train(tt)
        after = self.quantizer.reconstruct_n(0, len(ct))
        assert np.array_equal(after, ct), "native training changed frozen centroids"
        ids = np.arange(len(x), dtype=np.int64)
        self.index.add_core(len(x), faiss.swig_ptr(xt), faiss.swig_ptr(ids), faiss.swig_ptr(labels))
        for c in range(len(ct)):
            count = self.index.invlists.list_size(c)
            actual = faiss.rev_swig_ptr(self.index.invlists.get_ids(c), count).copy()
            assert np.array_equal(np.sort(actual), np.flatnonzero(labels == c)), "native list identity mismatch"
        self.index.max_codes = 0
        self.wrapper = faiss.IndexPreTransform(self.rotation, self.index) if self.rotation else self.index
        self.serialized_bytes = len(faiss.serialize_index(self.wrapper))

    def transform(self, x):
        return np.ascontiguousarray(self.rotation.apply_py(x) if self.rotation else x, dtype=np.float32)

    def search(self, q, probes, candidate_count):
        query = self.transform(q[None, :])
        coarse = distances(self.centers[probes], query[0]).astype(np.float32)[None, :]
        self.index.nprobe = len(probes)
        faiss.cvar.indexIVF_stats.reset()
        _, ids = self.index.search_preassigned(query, min(128, candidate_count), probes[None, :].astype(np.int64), coarse)
        assert np.all(ids >= 0) and len(np.unique(ids)) == ids.size
        return ids[0], int(faiss.cvar.indexIVF_stats.ndis)

    def metadata(self, raw_bytes):
        return {"kind": self.kind, "code_bytes_per_vector": self.index.code_size,
                "serialized_index_bytes": self.serialized_bytes,
                "rotation": "Faiss RandomRotationMatrix seed123" if self.rotation else None,
                "with_raw_refinement_storage_bytes": self.serialized_bytes + raw_bytes,
                "no_raw_refinement_storage_bytes": self.serialized_bytes,
                "serialized_sha256": hashlib.sha256(faiss.serialize_index(self.wrapper).tobytes()).hexdigest()}
