# Performance audit — gate closed

No ingestion/deletion speedup, QPS, recall-throughput frontier, or corrected-index
performance was measured in this campaign. T1–T3 correctness failures close that
gate; large datasets, k-means, multi-GPU work, and a repair are intentionally not
started.

At the pinned upstream revision, `hpdic/experiment/test_sivf_delete.cpp:46`
sets `baseline_time_ms = 202.2`. Lines 109–114 print that constant and divide it
by the measured SIVF time. This executable therefore does not itself provide a
fresh same-campaign Faiss deletion denominator. That source fact is not proof
that every result in the associated paper is invalid.

Before any future valid comparison: establish a passing correctness boundary;
remeasure the real Faiss GPU IVFFlat deletion path with identical semantics,
residency, quality target, workload, and timing scope; retain both natural and
adversarial deletion patterns, update success counts, pool conservation, and
all rejected measurements. A hypothetical Correct SIVF remains **not implemented**.

Nsight Systems and CUDA event timestamps in T4 are diagnostic evidence of
operation/kernel overlap, not benchmark performance numbers.
