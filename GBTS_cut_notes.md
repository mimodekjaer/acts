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
