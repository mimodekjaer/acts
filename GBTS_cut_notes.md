# GBTS cut / selection notes

Observations made while optimizing the GPU GBTS chain that concern the physics
selection. **No cut was changed** by the optimization work; everything below is
either a tunable capacity (no physics effect unless exceeded) or a note about
the previous implementation.

## Behaviour differences of the optimized code (all verified: identical seed
## totals on ttbar mu200, events 0-9)

1. **Old graph-making phi window lost candidates after a wrap-around node**
   (fixed, no cut change). In the pre-optimization `gbts_make_graph_edges`
   the per-thread sliding-window start (`last_n1`) was left at the top of the
   bin-1 chunk after processing a node2 whose phi window wrapped at +/-pi; the
   next (non-wrapping, low-phi) node2 of the same thread then broke out of the
   scan immediately and lost all its candidates. The rewritten kernel visits
   every candidate inside the exact +/- deltaPhi window (padded by 1e-5 rad,
   the exact `wrap_phi` cut still decides). The seed total went from 1 950 500
   to 1 953 000 (+0.13 %) on 500 events when this was introduced; the
   selection itself (cuts, windows) is unchanged.
2. **deltaPhi window computed on the device** (`min_delta_phi + dphi_coeff *
   maxDeltaR`) may differ from the host value by one float ulp (FMA
   contraction). No effect observed on the seed totals.
3. **Surface-to-layer lookup** for volumes spanning several layers now
   searches only the volume's own block (binary search); the old linear scan
   ran to the end of the whole map and could in principle have matched a
   surface index of a different volume. No effect on this geometry.

## New capacities (config fields, not cuts; exceeding them is deterministic
## truncation with a warning)

- `gbts_seedfinder_config::max_edges_per_spacepoint` (default 8): edge buffer
  capacity = factor x number of spacepoints (capacity, not the old
  `max_edges_factor` truncation on nodes). ttbar mu200 produces ~5.8
  edges/spacepoint (1.3 M edges for 223 k spacepoints), i.e. 27 % headroom.
  If a busier sample is used, raise it (memory: ~45 B x capacity, dominated by
  the neighbour lists).
- `gbts_seedfinder_config::max_connected_edges_per_spacepoint` (default 2):
  capacity of the compacted graph (edges kept after matching); observed
  ~0.45 / spacepoint.
- (Session B) `max_rows_per_connected_edge` (default 4): path-store capacity.
- The old `max_edges_factor` CLI option is gone (it truncated the edge set
  nondeterministically; nothing replaces it as a physics cut).

## Ideas that would change the selection (NOT applied)

- The phi window padding `gbts_phi_window_eps = 1e-5` only guards float
  rounding between the search and the cut; it is not a cut.
- The per-node cluster-width -> tau range (`gbts_sort_nodes_params`) and the
  RZ-doublet/curvature cuts (`gbts_make_graph_edges_params`) dominate the
  edge count (1.3 M edges of which only ~8 % survive matching). Tightening
  `max_Kappa_*` or the `z0` window would reduce the graph-making time roughly
  proportionally to the candidate count, but that is a physics decision; no
  change recommended from the performance side.

## Capacities that can drop data (Session B)
Not physics cuts, but limits that truncate deterministically when exceeded
(a warning is printed at the start of the next event):
- `max_rows_per_spacepoint` (2): path-store rows (typical need ~0.35/spacepoint).
- Resident CCA grid: the fused CCA handles at most
  (resident blocks x 1024) connected edges, ~270k on an H100 (typical: 100k);
  edges beyond are dropped from seeding (`nCcaDropped` counter).
- Seed output capacity: 2 seeds per path-store row.

## CCA terminus flag race (FIXED, selection change in racy cases)
In gbts_run_cca_iteration an edge that settles writes
`outgoing_paths[nei].y = -1` for all its neighbours ("not a terminus"),
while a neighbour that settles in the same iteration writes its own
`outgoing_paths[nei] = {out_paths, terminus flag}`. The two writes race; the
surviving value decides whether `nei` is a terminus edge (a seed path root).
This is present in the original per-iteration kernel too, so results depend
on the GPU's execution schedule: the same code with 1024-thread blocks and a
different grid produced 1 642 679 instead of 1 608 480 seeds on events 5-9.
FIXED (optimization log entry 15): the neighbour mark now goes to a separate
`has_parent` byte array and terminus = `settled && !has_parent`. In the racy
cases the old code kept whichever write landed last; the new rule is the
deterministic union (an edge that is any settled edge's neighbour is never a
terminus). This IS a selection change relative to some schedules of the old
code - the seed totals moved (events 0-9: 781 200 -> 782 940 @ 200
processed; events 5-9: ~1 608 480 -> 1 608 480x/804 240 @ 200) - but the old
totals were themselves schedule-dependent, so there was no well-defined
previous selection to preserve. Physics owner should re-validate efficiency
once on the new deterministic baseline.

## Residual seed-bidding nondeterminism (+/-1 seed) - RESOLVED (session C, entry 16)
Not a bidding race: the upstream GPU clusterization/spacepoint formation
delivers the spacepoints in a run-dependent order, and the node sort broke
exact-phi ties by spacepoint index. The tie-break now uses (phi, r, z,
width) before the index; the whole chain is reproducible (verified array by
array). Note for validation: seeds must be compared by spacepoint
COORDINATES, the spacepoint indices in the seed output follow the
non-reproducible upstream order. The original text of the investigation is
kept below for reference.

## (superseded) Residual seed-bidding nondeterminism (+/-1 seed, OPEN)
After the fixes above one flip remains: repeated runs give 804 240 vs
804 200 total track parameters on events 5-9 @ 200 processed (i.e. +/-1
seed in one event per 40-event cycle) and 782 940 vs 782 960 on events 0-9.
Bisection evidence:
- Per-stage counters (edges, connections, connected edges, fit proposals,
  rejected proposals) are bit-identical across runs -> the graph, CCA,
  path store and fit are deterministic; the flip is a SWAP of which
  proposals end up rejected (same count), which changes the final total
  only through the converter's dropout step emitting 1-2 seeds per
  accepted proposal.
- With 0 bidding rounds the totals are exactly stable.
- The flip appears in BOTH the fused finish kernel and the default
  (separate-launch) bidding path -> the race is in the shared bidding
  device code, not in the fusion.
- Separating first-round classification into its own phase did not
  remove it.
Suspects (not yet proven): in `gbts_reset_edge_bids.ipp` the winner check
`d_seed_ambiguity[best_bid & 0xFFFFFFFF] == 0` dereferences seed 0 when a
path edge carries `best_bid == 0` (no bid), and rows whose proposal was
never created (`prop.y < 0`) keep ambiguity 0 and can masquerade as
"unmarked winner"; also the -1 "maybe" marks written by
`create_seed_candidate` while other threads read the same flags within one
round. A clean fix probably needs the reset pass to double-buffer the
ambiguity flags (read round N, write round N+1) the way the bids already
are. Left open for the physics/algorithm owner; effect size is 1 seed in
~800 000.

## Seed bidding rounds (session C, entry 19) - PLEASE REVIEW
The seed-vs-edge bidding rounds (`edge_bidding_rounds`, default was 5)
cannot change the seed selection with the current device code: after the
classification (0 -> 1 "maybe", else -2 "rejected") only 1/-1 proposals bid,
and the reset step rejects a proposal only if an edge of its path is held
by a proposal with ambiguity 0, which no bidding proposal has. Verified:
identical seeds for 0, 1 and 5 rounds on 15 events; default set to 0.
If the intended algorithm is the CPU one ("clean winners keep 0 and do not
rebid; a maybe is rejected when one of its edges is held by a clean
winner"), the port would need: (a) the classification to leave clean
winners at 0 (and exclude them from rebidding), (b) the reset test as it
is. That WOULD change the selection (more rejections); it is a physics
decision, not made here.
