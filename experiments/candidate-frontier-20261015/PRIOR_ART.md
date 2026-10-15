# Gate 0: nearest mechanisms and baseline identity

Verified from primary papers and repositories on 2026-10-15. This is direction
triage, not an exhaustive literature review or a new-method claim.

| Primary source | Established overlap | Consequence for this screen |
|---|---|---|
| [RaBitQ, SIGMOD 2024](https://arxiv.org/abs/2405.12497) | Low-bit high-dimensional distance estimation with an error bound | Quantization and error-aware filtering are not new here. |
| [Extended RaBitQ, SIGMOD 2025](https://arxiv.org/abs/2409.09913) | More-bit space/error trade-offs | Compare multiple data-code precisions, not only a one-bit strawman. |
| [GPU-native IVF-RaBitQ](https://arxiv.org/html/2602.23999v1) | One-bit filtering, extended-code refinement, fused local top-k and GPU layouts; can avoid raw-vector reranking | Beating FP32 Flat in bytes is not novelty or evidence of beating this GPU baseline. |
| [ADSampling](https://arxiv.org/html/2303.09855v1) | Adaptive partial-distance decisions, including probabilistic comparison | Ordinary partial-dimensional pruning is a known control, not a contribution. |
| [DARTH, SIGMOD 2026](https://arxiv.org/abs/2505.19001) | Query-level recall-aware early termination | A different axis from candidate certificates. Source-audited only; not executed here. |
| [TurboQuant, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) | Low-cost high-dimensional vector quantization | A new representation must distinguish itself from established low-bit information. Not executed here. |

## Exact source identities

- CPU controls: [Faiss v1.15.1](https://github.com/facebookresearch/faiss/tree/v1.15.1),
  commit `75c755c5a1fc854756aaf3bd1131213aeb21d8d1`.
  [Native RaBitQ tests](https://github.com/facebookresearch/faiss/blob/v1.15.1/tests/test_rabitq.py)
  cover rotations and extended bits. Actual library classes, not replicas, are
  used for Flat, PQ and RaBitQ. Query bits=4; data bits=1 or 4.
- GPU artifact: [cuvs_rabitq](https://github.com/Stardust-SJF/cuvs_rabitq),
  branch `cuvs_ivf_rabitq`, commit `33d1de70fba2bd86390ea5915c62917c3a85c2ff`.
  This is the repository reached from the paper's artifact link. Source inspected;
  no build, GPU execution, timing or exact GPU-baseline reproduction in this stage.
- DARTH: [public code](https://github.com/MChatzakis/DARTH),
  commit `0d9bafcf31d1d79668bc71139fe93fa5e70b5185`. Its documented build uses
  CPU Faiss plus LightGBM. Do not compare CPU QPS against GPU QPS.

## Novelty kill test

All five certificates in this screen are known lower-bound constructions.
The SQ8 result cannot be promoted as a new thesis: it is standard reconstruction
plus residual error, still computes a D-dimensional distance, and is already
weaker in this logical-byte model than several native quantization controls.
A GPU timing gap would not by itself rescue an otherwise subsumed mechanism.
No new non-incremental mechanism has been demonstrated. Missing GPU measurements
are uncertainty, not evidence that every possible candidate-information mechanism
fails. A future direction must identify what information or execution work it
removes beyond the nearest GPU-native baseline before another expensive campaign.
