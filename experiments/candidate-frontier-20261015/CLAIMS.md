# Claim–evidence ledger

All rows refer to experiment `ivf_20261015_candidate_information_cpu_v1`, frozen
CONTRACT.md, finite prefixes and CPU execution on 2026-10-15. Evidence state is
not a claim that the entire research brief has been completed.

| ID | Claim and denominator | State | Evidence / oracle | Counterevidence and allowed wording |
|---|---|---|---|---|
| C1 | Oracle opportunity across 64 held-out queries per dataset | `partial` | QUERY.jsonl.gz perfect_answer_oracle; free selector | Large ideal space only; not a realizable filter or speedup. |
| C2 | Five guarded online controls preserve selected-list exact IDs in every evaluated case | `measured` | Primary assertions plus independent replay receipts | Finite matrix only; mathematical construction is not a universal floating-point proof. Global recall remains routing-limited. |
| C3 | SQ8 online read ratio is .28676/.27108/.27573 at calibrated probes | `partial` | Explicit logical read ledger, refetch model | Query overhead, cache/transactions, time, encoding and workspaces incomplete; no physical-byte claim. |
| C4 | Known native quantization controls are a stronger logical-byte counterexample on real subsets | `partial` | Native 4-bit RaBitQ+32, all 64 answers per real subset identical to candidate-exact GT | Not a general exact guarantee, not native GPU execution, not recall-tuned; no speed ranking. |
| C5 | Grain expansion removes modeled savings | `partial` | Recorded grain 1/8/32 proxies and mask hashes | No measured warp/cache behavior; cannot assert H3 as the true GPU bottleneck. |
| C6 | Useful query-difficulty adaptation or GPU Pareto advantage | `unknown` | No GPU timing or scheduling evidence | Gap labels are evaluation-only; batch proxies are insufficient. No positive or negative GPU-performance claim. |
| C7 | New research mechanism | `unknown` | Known constructions; nearest-art overlap in PRIOR_ART.md | No novelty established; stop the current implementation path rather than package it as a thesis. |

Exact current source/input/results hashes are bound in MANIFEST.json and RUN.json.
Off-Git models and raw-log hashes are retained in RUN.json; model training time
is observation-only. Independent checks distinguish x86 native saved-model
replay from arm64 real-data certificate replay. No cross-architecture model
training or timing equivalence is claimed.
