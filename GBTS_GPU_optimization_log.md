# GBTS GPU optimization log

Branch: `feat/gbts-traccc-determinism`. Hardware: NVIDIA H100 NVL, CUDA 13.1,
build `cuda-fp32` preset (`CMAKE_CUDA_ARCHITECTURES=75`, PTX JIT on the H100).
Data: `/shared/traccc/data/ttbar_mu200`, ITk geometry, GBTS config in
`/shared/traccc/GBTS`.

Measurement protocol for every entry:
- Throughput: `traccc_throughput_mt_cuda ... --processed-events=500
  --reco-stage=seeding-only` (only the GBTS seeding is timed; clusterization
  and spacepoint formation are pre-computed), run twice; the two seed totals
  ("Reconstructed track parameters") must be identical (determinism check).
- Latency: `ncu --metrics gpu__time_duration.sum -k regex:"gbts|cub"` over 3
  events; per-kernel time summed per event.

Constraint: the physics cuts and the produced seeds must not change (bit-level
differences caused by reordering float arithmetic are documented per entry).

## Baseline (commit 26f710ead)
- seeding-only: 1.030 ms/event (970 ev/s), 1 953 000 seeds, deterministic.
- GPU busy per event ~813 us, GPU idle ~315 us (nsys): idle is host-side:
  gaps before memsets (27/event, 94 us), before D2H readbacks/waits (77 us),
  before H2D uploads from pageable memory (10/event, 36 us).
- Top kernels (us/event): make_graph_edges fill 162 + count 113, CCA 15x7.4
  = 112, match 76, bin_spacepoints 53, rebid 5x10.5 = 52, reset 5x8.4 = 42,
  node radix sort ~80, fill_path_store 20, compress 17.
- ncu: make_graph_edges and match are instruction bound (issue active
  70-78%, SM throughput 42-71%); bin_spacepoints (issue 4%, 0.8 waves,
  single-address atomic cursor) and CCA (issue 11%, 0.4 waves) are latency /
  launch bound.

## Entries

### 1. Pinned static tables + memset elimination (FASTER, small)
Result: 1.030 -> 1.010 ms/event, seeds identical.
- Static tables copied per event from pageable std::vectors -> pinned
  vecmem::vector members (async copies no longer stall the host thread).
- 10 of 16 explicit per-event memsets removed by letting kernels initialise
  what they own (kept flags in the fill pass, neighbour counts in match,
  CCA levels in compress, seed proposals/ambiguity in fill_path_store, edge
  bids in count_terminus, next-round bids in reset_edge_bids via a double
  buffer).
Remaining memsets: eta_node_counter, num_incoming_edges, hit_bids, counters.

Baseline numbers (commit 26f710ead, rebuilt): seeding-only throughput
**1.06 ms/event** (943 ev/s), GBTS GPU kernel time 844 us/event, GPU idle
inside the seeding ~320 us/event (11 host syncs, ~60 launches/memsets/copies).
Kernel ranking (us/event): make_graph_edges fill 162, count 112, CCA 15x7.4 =
111, node radix sort (8 one-sweep passes of 64-bit keys) 112, match 76,
bin_spacepoints 54, rebid 5x10.5 = 52, reset_edge_bids 5x8.4 = 42,
fill_path_store 20, compress 18, convert 14, fit 9, ...

### 1. Drop debug-only device counters and an unneeded memset  -> 1.016 ms/event (-4%)
- `nConnections` (match kernel) and `nTerminusEdges` (count_terminus) were
  same-address global atomics feeding only DEBUG log lines / a redundant
  zero check; removed together with the counters D2H copies.
- The `neighbours` buffer memset was unnecessary (only the first
  `num_neighbours[e]` entries of a row are ever read).
Effect: -2 contended atomics per edge/terminus, -1 memset, -2 D2H copies.
Determinism: identical seed totals (1 953 000 / 500 events).

### 2. Asynchronous node stage  -> 0.995 ms/event (-2%)
- `gbts_bin_spacepoints` writes every sort key in spacepoint order (rejected
  spacepoints / unused capacity get a key that sorts last); no more slot
  atomic. The node count and the per-eta-bin node ranges are produced on the
  device by the new single-block `gbts_build_edge_work_list` kernel (strip
  scan, one block scan per phase), which also lays out the graph-making work
  items `(pair, chunk)`. All node/edge buffers are sized by the spacepoint
  capacity; `make_graph_edges` grabs work items dynamically (one atomic per
  block and item, output positions fixed -> deterministic).
- Removes the `eta_counts` D2H wait, the host work-list loop and 2 H2D copies.
- Node sort: `cub::DeviceRadixSort::SortPairs` on the 44 significant key bits
  (eta bin | phi) instead of a 64-bit `thrust::sort_by_key`: 6 one-sweep
  passes instead of 8 (radix sort stability provides the spacepoint-index
  tie break, since the keys are written in spacepoint order).
Tried and rejected on the way: static block-striding over the work list with
a per-block binary search for the pair (+25 us per pass: 12 dependent loads
before any work), bigger grids (worse tail), Hillis-Steele scan per 512
elements in the work-list kernel (~100 barriers, 13-20 us -> strip scan).
Costs: `bin_spacepoints` 53 -> 70 us (writing keys for the full capacity;
this kernel is a latency-bound chain of ~6 dependent loads and swings with
buffer placement), `make_graph_edges` +10 us per pass from the dynamic work
grabbing. Net GPU-kernel time 849 -> 890 us but ~70 us less host/sync gap.
Determinism: identical seed totals.

### 3. Asynchronous edge stage  -> 0.942 ms/event (~neutral, -1 host sync)
- Edge buffers sized by a capacity `max_edges_per_spacepoint * nSp` (new
  config field, default 8; ttbar mu200 produces ~5.8 edges/spacepoint); the
  fill pass truncates deterministically at the capacity and stores the
  device-side edge count + the uncapped total (warning after the stage's one
  synchronisation if edges were dropped). match / re-index / compress read
  the edge count on the device (grid-stride loops with capped grids); a one
  thread kernel stores the kept-edge count after the re-index scan, so the
  stage synchronises once (kept count -> compacted graph allocation) instead
  of twice. The initial `get_size(spacepoints)` sync is replaced by the
  collection capacity as well.
- The kept flags are 1-byte (scanned into the int re-index through a
  transform iterator): the scan over the capacity (8x nSp instead of the
  exact 6x nNodes) costs 12.7 us instead of 8.9 us with int flags over the
  exact count (17.7 us with int flags over the capacity).
Net: removed 2 host syncs, +~10 us of kernel time -> 0.945 -> 0.942 ms/event.
Determinism: identical seed totals.

## Session B (second Claude session, worktree branch, cherry-picked onto the branch)

Measurement protocol as above; every entry: seeding-only ms/event over 500
events, two runs with identical seed totals (1 953 000), ncu over 3 events.

### B1. Fused cooperative CCA + seed bidding, two fewer host syncs (FASTER)
1.010 -> 0.963 ms/event. The 15 CCA iterations run in one cooperative kernel
(grid.sync between iterations); the initial bid + 5 rebid/reset rounds in
another. nProps/nRejected readbacks removed (later kernels loop over the rows;
seed output sized by the row count). Kernel time itself barely changed
(CCA 111 -> 114 us, bidding 100 -> 92 us): the win is the removed launch gaps
and syncs.
- 1024-thread blocks instead of 128: CCA 100 -> 87 us, bidding 92 -> 70 us
  (the grid-wide barrier cost scales with the number of blocks).
- Neighbour lists cached in registers across iterations + early exit when no
  edge is active: CCA 114 -> 100 us. Active edges per iteration (event 0):
  61k, 21k, 14k, 10k, 7k, 4.7k, 2.9k, 1.6k, 769, 329, 110, 29, 6, 0 of 101k,
  i.e. from iteration 4 on the cost is the fixed per-iteration latency
  (level loads + barrier), not work.

### B2. Register-cached fused bidding (FASTER, small)
Bidding kernel 70 -> 65 us: each row's proposal chain (<= 16 edges) is walked
once and kept in registers across the rounds.

### B3. Things that did NOT help (kept out)
- Block-aggregated write cursor in gbts_bin_spacepoints: 53 -> 53 us. The
  compiler already warp-aggregates the single-address atomic.
- `#pragma unroll` of the chain loops (bidding atomics, CCA neighbour level
  loads) for memory-level parallelism: no measurable change.
- CCA "tail mode" (after iteration ~4 only 8 blocks continue over a compacted
  active list with a custom spin barrier, the other blocks retire):
  SLOWER, 87 -> 120 us. The per-iteration compaction (ballot/scan/extra block
  barriers) and the spin barrier cost more than the grid barrier they replace.

### B4. Block-cooperative outer-node range search in make_graph_edges (FASTER)
0.945 -> 0.927 ms/event (count 125 -> 110 us, fill 177 -> 167 us). Every
thread used to redundantly run the block's 2-4 global binary searches (18
steps each) per work item, ~30% of the kernel's instructions. Now the block
probes 128 evenly spaced nodes and 32 threads per boundary refine inside the
probe interval (two rounds, four barriers).
Cut selectivity measured in the count pass (event 0, 7.2M candidates):
dr 6%, tau bounds 24%, z0 47%, zouter 0%, dphi ~0% (window pre-selection),
curvature 5%; ~17% accepted. Reordering the cuts cannot buy much: everything
after dr needs the tau division anyway.

### (rejected) Per-eta-bin segmented node sort
Scatter the nodes into their eta-bin segments (eta counters used as
cursors) and `cub::DeviceSegmentedSort` the segments by a unique
(phi bits, spacepoint index) key. Result: the scatter kernel alone costs
65 us (consecutive spacepoints share an eta bin -> same-address atomic
contention) and the segmented sort falls back to per-segment radix sorts for
the large inner-layer bins (70 us): 136 us vs 96 us for the 44-bit global
radix sort. Reverted.

### 4. 32-bit node sort keys  -> 0.910 ms/event (-0.7%)
Node sort key = (eta bin << 20) | 20-bit quantised phi, sorted with
`cub::DeviceRadixSort` on the significant bits (4 one-sweep passes instead
of 6; keys and key traffic halved). Determinism and the exact phi order are
preserved: `gbts_sort_nodes` detects runs of equal keys (phi within
2*pi/2^20) and places their nodes by the exact (phi, spacepoint index) rank,
so the per-eta-bin node order is the exact phi order as before. Seeds
identical. Node sort 96 -> 89 us (the one-sweep passes got slower per pass
with 32-bit keys, 14 -> 20 us, so the gain is smaller than 6/4 would give).

### (rejected) Two small ones
- `gbts_build_edge_work_list`: writing the work items by all threads with a
  binary search of the owning pair instead of by the pair's owner thread:
  20 -> 24 us (the dependent global loads of the search cost more than the
  serialised stores of the few large pairs).
- `gbts_compress_graph`: testing the 1-byte kept flag instead of comparing
  two scan values: no measurable change (+-1 us). Reverted both.

### 5. One static-table upload and one memset per event  -> 0.874 ms/event (-4%)
All static tables (volume/surface layer maps, layer type/info/geo, tau LUT,
bin pairs, pair group begins) are packed once into a single pinned host
blob (16-byte aligned sections) and uploaded with ONE H2D copy per event
(was 8); the named counters, the eta node counters and the edge CSR share
one zeroed `unsigned int` buffer (one memset instead of three). No kernel
change: the gain is purely fewer GPU operations (each memcpy/memset costs
~2-3 us of launch gap on the timeline). Seeds identical.

### 6. Binary search in the surface-to-layer map  -> bin_spacepoints 70 -> 62 us
`setLinkingScheme` writes the surfaces of a multi-layer volume as a block
[(count, 0), (surface index, layer)... sorted]; `gbts_bin_spacepoints`
binary-searches the block instead of scanning linearly (previously up to the
end of the whole 1140-entry map). Seeds identical; throughput 0.874 ->
0.873 ms/event (the kernel is otherwise a latency-bound chain of dependent
loads: spacepoint -> measurement -> surface link -> volume map -> layer).

### 7. No re-index "finish" kernel  (neutral, -1 launch)
The fill pass zeroes the kept flags beyond the edge count (every thread of
the grid, after the work loop), so the prefix sum over the whole capacity
ends with the kept-edge count and the host reads it back directly together
with the counters; the one-thread `gbts_reindex_edges_finish` kernel and its
launchers are gone. Throughput unchanged within noise (0.863 ms/event).
Lesson recorded: the first version zeroed inside the work-item loop, which a
block that grabs no work item never executes -> garbage count in the MT
example (where block timing differs); the seq example passed by luck
(fresh, zeroed memory).

### (rejected) min/max radius accumulated in gbts_sort_nodes
Atomic min/max on the radius bits per eta bin from the node sorting kernel
(to drop the gbts_find_minmax_radius launch): the nodes are sorted by eta
bin, so all lanes of a warp hit the same two addresses -> sort_nodes 7 ->
86 us. Reverted (the separate 8 us block-reduction kernel stays).

### B5. Debug seed-count readback removed (FASTER)
0.917 -> 0.907 ms/event: copy().get_size(output_seeds) at the end of
extract_seeds was a full synchronisation used only for a debug message.

### B6. Device-side path-store row count (FASTER)
0.865 -> 0.852 ms/event (on the peer's 46955cdd8/47280b79a). The nRows
readback (and the ~15 us launch gap after it) is gone: the path store /
proposals / seed output get a capacity of max_rows_per_connected_edge (4) x
nConnectedEdges (typical events need ~0.8 rows per edge), the tail kernels
grid-stride to a device-side row count and are launched with grids sized for
nConnectedEdges. hit_bids zeroing moved into the terminus kernel (one memset
less). Rows beyond the capacity would be dropped (documented in the config).
The fused bidding kernel caches one row per thread and processes rows beyond
the grid uncached, so it stays correct for any row count (66 us).
Tried on the way: 2 or 4 cached rows per thread with the grid sized for the
capacity -> chains spill to local memory and rows serialise: 80 us, i.e.
SLOWER than the uncached fused kernel (70 us); dropped.

### B7. Other things that did NOT help (kept out)
- 512-node outer slabs in make_graph_edges (fewer barriers): no change.
- Shift instead of the 64-bit multiply/divide for the probe positions of the
  cooperative search: no measurable change (the kernel is not purely
  instruction bound after B4).
- Bare cost of the 15 grid barriers in the fused CCA is 30 us (measured with
  the iteration body disabled); the remaining ~55 us are the level loads of
  the first iterations, i.e. the CCA is close to its floor with this
  algorithm.

### B8. Kernel fusions in the seed tail (FASTER, small)
- Hit bidding and seed conversion appended as two more grid-synchronised
  phases of the cooperative bidding kernel (gbts_finish_seeds_kernel launcher
  virtual, CUDA override; other backends keep three launches):
  0.852 -> 0.848 ms/event (81 us fused vs 66 + 9 + 13 us plus two launch gaps).
- Segment fit folded into gbts_fill_path_store: the descent that lays out a
  row already visits its root-to-leaf edge chain, which is the reversed
  sequence the Kalman fit walks through the parent links, so the fit runs
  on the chain kept in registers (bit-identical operations). Removes the
  fit_segments kernel: 0.848 -> ~0.842 ms/event (fill+fit 24.6 us vs 21 + 9).
Seeds identical (1 953 000) in every step.
