# Supplemental probe-grid calibration

The original contract uses nprobe 1/2/4/8/16/32/64. After inspecting primary_02,
we observed examples where the minimum listed probe meeting 0.90 recall jumps
from 4 to 8 and overshoots recall to about 0.98. A work-count increase at that
coarse-grid point cannot all be attributed to index degradation.

Keep the original grid, raw answers, estimator and results unchanged. Add an
explicitly exploratory all-integer nprobe 1..64 curve, before inspecting its
values, to diagnose this competing explanation. This is not a new acceptance
rule or a performance measurement. For exact top-10 inside a selected list set,
Recall@10 equals the fraction of global top-10 IDs whose list is selected, using
the same (distance, ID) tie rule. Derive recall from those ten list ranks and
work counts from the cumulative per-list live/extent counts. Preserve both in
raw query records. Assert that every original-grid result equals its matching
all-integer point. Physical returned-ID checks remain on the original grid.

This supplement was introduced after observing primary_02, not preregistered
with the original experiment. Do not present a coarser-grid amplification as an
iso-recall lower bound or any scan count as latency/DRAM traffic. Values from
primary_02 are not release-admitted: a later audit exposed a missing sparse-ID
uniqueness assertion, although no generated-trace duplicate was established.
Primary_03 adds that assertion and a cause-specific mutation test; its source
and observations are preserved before adding this supplement.
