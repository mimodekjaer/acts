# traccc seeding-stage performance log (clusterization / spacepoints / track parameters)

Branch: `perf/traccc-clustering` (acts monorepo, `Traccc/` subtree). Started 2026-09-02 18:16 CEST.

Goal: make the CUDA "seeding" throughput stage (H2D copy, clusterization, measurement
sorting, spacepoint formation, [GBTS seeding, untouched], track-parameter estimation,
D2H copy) as fast as possible with *identical* clusters, spacepoints and parameters,
deterministic output, and portable (common device code, all backends).

Benchmark command: the `traccc_throughput_mt_cuda` command from the task (2000 events,
ttbar mu200, ITk, GBTS, `--reco-stage=seeding`, 1 thread). GPU: NVIDIA H100 NVL.
Validation: `traccc_dump_seeding_cuda` (new helper app, `Traccc/examples/run/cuda/apps/`)
dumps measurements / spacepoints / seeds / params with full precision; `cmp.py` compares
two dumps both order-sensitively ("raw") and as multisets ("canonical").

## Baseline (commit 4502b8795, unmodified)

| run | ms/event | events/s |
|-----|----------|----------|
| baseline1 | 7.853 | 127.3 |

### Determinism of the baseline: NOT deterministic
Two runs of the dump tool on events 0-4 give identical *sets* of measurements and
spacepoints, but a different *order* of measurements within a surface (the CCL kernel
assigns output slots with an atomic `push_back`, and the measurement sorting only sorts by
surface id). The GBTS seeding is sensitive to the spacepoint order, so the seed and
track-parameter sets differ between runs (e.g. event 0: 3979 vs 3994 seeds).
=> A deterministic measurement order is a prerequisite for the required determinism.

## Attempt 1: radix-based, deterministic measurement sorting (CUDA) — SUCCESS

Profile of the baseline (nsys, 300 events): 73% of GPU kernel time was the
`thrust::sort` of the measurement indices with the `geo_id_based_sorter` comparator
(merge sort over the *capacity* = number of cells, ~1M elements, gathering 64-bit
surface ids for every comparison): ~5.0 ms/event. CCL kernel: 0.27 ms, GBTS: ~1 ms,
H2D copy of cells: ~0.4 ms, spacepoints/params: ~0.05 ms.

Change (commit "Deterministic radix-based measurement sorting on device"):
* CCL kernel (`aggregate_cluster.ipp`, common code): `identifier` = index of the
  cluster's first (root) cell — unique and deterministic, unlike the atomic slot.
* New common device functions (`clusterization/device/measurement_sorting.hpp`).
* CUDA `measurement_sorting_algorithm`: get the measurement count on the host (async
  size), then `thrust::stable_sort_by_key` twice on primitive keys (radix sort):
  first by the cluster key (32 bit), then by the surface id (64 bit). Output buffer is
  sized exactly; output `identifier` = `cluster_index` = position (host semantics).

| run | ms/event | events/s |
|-----|----------|----------|
| v1_sort1 | 2.838 | 352.4 |

Validation (dump, 5 events, two runs): measurement lines bit-identical between runs
(deterministic) and the measurement/spacepoint *sets* identical to the baseline.
Spacepoints are still in a non-deterministic order (atomic `push_back` in
`form_spacepoints`), so seeds/params still differ between runs -> Attempt 2.

Note on the profile: the ~480k tiny H2D copies and ~120k pinned allocations seen in
the profile happen at construction time (jagged copies of the detector description,
done for every `full_chain_algorithm` copy), not per event. They do not affect the
"Event processing" time.
## Attempt 2: deterministic spacepoint formation (flags + prefix sum + scatter) — SUCCESS (determinism)

`form_spacepoints` used `push_back_default` (atomic slot), so spacepoint order was
non-deterministic. Now: flag valid measurements -> inclusive scan (thrust / oneDPL /
alpaka wrappers) -> D2D copy of the last prefix sum into the collection size -> scatter
each spacepoint to `prefix_sum - 1`. Measurement and spacepoint dumps are now
bit-identical between runs (0 differing lines, 5 events). Ported the sorting and this
change to Alpaka / SYCL / HIP (not compiled here: only CUDA is built in this checkout).

Seeds and track parameters still differ between two runs on *identical* spacepoints
(e.g. event 0: 3979 vs 3997 seeds), so the remaining non-determinism is inside the
GBTS seeding algorithm itself (not to be changed per the task). Track-parameter
estimation is one thread per seed and deterministic given the seeds.

| run | ms/event | events/s | note |
|-----|----------|----------|------|
| v2_sp (time-seeded random event order) | 3.015 | 331.7 | vs 2.838 for v1: noise or scan cost? see fixed-seed runs |
## Attempt 3: fixed-size sorted measurement buffer — SUCCESS (neutral within noise)

The sorted measurement buffer is now non-resizable (its size is known on the host), so
`copy().get_size()` in the spacepoint formation no longer synchronises with the device.
Fixed seed (`--random-seed=12345`) A/B runs: v2 2.77-2.81 ms/event, v3 2.72-2.85 ms/event
(noise ~ +-3%). Kept: fewer host syncs, identical output.

## Attempt 4: config sweep `--target-cells-per-thread` (no code change)

ncu on the CCL kernel (default config, 256 threads x 8 cells): 437 blocks = 0.55 waves
on the H100, 63% of cycles with no eligible warp, 40 registers, 16 KB shared memory:
latency bound with too little parallelism. Sweep with the fixed seed:

| target cells/thread | ms/event |
|---|---|
| 8 (default) | 2.72 / 2.85 |
| 4 | 2.65 |
| 2 | 2.66 |
| 1 | see log (sweep_tc1) |
| 16 | see log (sweep_tc16) |

Partition sizes never exceed 4096 for target >= 2 in ttbar mu200 (max 2205 for 2048
target), so the slow global-memory fallback path is never taken.

## Attempt 5: restructured CCL: parallel partition search, prefix-sum slots, separate aggregation kernel — SUCCESS

Emulating the partition boundary walk of the CCL kernel (thread 0 walking cell by cell
with dependent global loads until a module boundary / channel1 gap) on event 0 gave
walks of up to 188-256 cells (p99 ~55), i.e. tens of microseconds of serial latency per
affected block. Changes (all in common device code, wrappers per backend):

* `ccl_kernel`: partition boundaries found with a parallel block search
  (`find_partition_split`: every thread tests one candidate, shared-memory atomic
  min; identical result to the sequential walk). The kernel no longer creates
  measurements: it writes a root flag per cell and the cluster linked list in global
  cell indices (`next_cell`), 8 bytes per cell.
* backend launcher: inclusive scan of the root flags (thrust / oneDPL / alpaka wrapper),
  D2D copy of the last prefix sum into the measurement collection size, then
  `aggregate_clusters` (one thread per cell; root threads walk their list and write the
  measurement at `prefix_sum - 1`). The slot order is now deterministic (cell order),
  so the measurement sorting needs only one stable radix pass (by surface id).
* identifier == index of the first cell (unique, deterministic); cluster_index == slot;
  the disjoint-set / cluster-size outputs are written by the aggregation kernel.

Validation: measurement and spacepoint dumps bit-identical to attempt 3 (0 differing
lines). CUDA clustering / spacepoint unit tests pass. (The 640 `*cca*` tests fail only
because the `data/cca_test` files are not present in this checkout.)

| run | ms/event |
|-----|----------|
| v4_s1 / v4_s2 (default target 8) | 2.658 / 2.714 |
| v4_tc4 (target 4) | 2.648 |

## Attempt 6: single radix pass in the measurement sorting — SUCCESS

With deterministic CCL slots the pre-sort by cluster key is unnecessary: one stable
radix pass by surface id (CUDA/HIP). Dumps bit-identical to attempt 3.
(A first benchmark pair gave 3.58-3.60 ms/event, but the 1.3 GB traccc test-data download
and unpack were running concurrently; repeated afterwards.)

| run | ms/event |
|-----|----------|
| v5_s3 / v5_s4 (fixed seed) | 2.680 / 2.591 |

Profile after attempt 6 (nsys, 300 events, fixed seed): kernel time 1.57 ms/event.
`ccl_kernel` 116 us (was 267), `aggregate_clusters` 97 us, radix sort 96 us + 15 us
gather, spacepoint flags/scan/form ~30 us, `estimate_track_params` 43 us, GBTS ~1.1 ms
(untouched), H2D copy of the cells ~0.37 ms (5 x 4 MB, pinned, PCIe bound).
The 640 CUDA CCA tests (with the downloaded `traccc-data-v10`) pass, including the
"WithScratch" variant that forces the global-memory fallback path.

Benchmark noise: from ~00:55 other build jobs of the same user (`test_dev/gbts_changes`,
two `-j20` builds) load the machine (load average 30-50); identical binaries then measure
between 2.59 and 2.92 ms/event. Numbers below are the best of several runs unless noted.

## Attempt 7: skip the sort when the measurements are already sorted — (see below)

The cell reader groups cells by ascending geometry id, so the deterministic CCL output
is already sorted by surface in practice. A small kernel flags any out-of-order
neighbours; the flag is read back together with the size (one synchronisation). If the
input is sorted, the output is produced by a plain copy kernel (identifier/cluster index
= position, exactly what the stable sort would produce). Implemented for CUDA, HIP,
Alpaka and SYCL.

| run | ms/event | load avg |
|-----|----------|----------|
| v7_s1 / s2 / s3 (fixed seed) | 2.522 / 2.564 / 2.522 | ~33 |

Dumps bit-identical (0 differing lines). SUCCESS.

## Attempt 8: upload only the cell columns used by the clusterization — SUCCESS

The cells are uploaded as 5 columns x 4.2 MB (u32 channel0, channel1, f32 activation,
f32 time, u32 module_index) at ~56 GB/s = ~380 us/event, i.e. ~15% of the event time.
The `time` column is never read on the device, so the full-chain algorithms (CUDA, Alpaka,
SYCL) now copy only the 4 used columns (`examples/.../copy_cells.hpp`). The buffer keeps
its layout; the time column is simply left uninitialised and never read.

## Attempt 9: one warp per block for the seed parameter estimation — SUCCESS

ncu: `estimate_track_params` ran 27 blocks x 128 threads (~3.4k seeds), 98% of cycles
with no eligible warp, i.e. a latency-bound kernel using 27 of 132 SMs. With 32-thread
blocks (all backends) the same work spreads over ~108 SMs. Results are per-thread and
unchanged (params identical on the seeds common to two runs).

Wall-clock A/B is very noisy at this point (load average 30-50 from other jobs plus a
low-priority baseline build for validation): v8 = 2.46 / 2.93 / 2.71 ms/event. GPU-side
effect measured with nsys instead (see below).

Profile after attempt 9 (nsys, fixed seed): kernel time 1.443 ms/event; `estimate_track_params`
23 us (was 46), measurement "sort" = 13 us copy kernel, cell upload 4 columns.

## Attempt 10: CCL launch configuration sweep by kernel time (nsys, 100 events) and new default

| config | ccl_kernel | aggregate_clusters | kernels/event |
|---|---|---|---|
| 256 threads x 8 cells (old default) | 115.5 us | 96.7 us | 1.443 ms |
| 256 x 4 | 100.0 us | 96.5 us | 1.428 ms |
| 256 x 2 | 100.8 us | 95.7 us | 1.427 ms |
| 128 x {4,8}, 512 x {2,4} | runs failed (see cfg_*.log) | | |

=> default `target_cells_per_thread` changed 8 -> 4 in `clustering_config.hpp` (identical
results, partition boundaries do not affect cluster content). SUCCESS (-15 us/event).

## Attempt 11: one thread per cluster in the aggregation (root compaction)

`aggregate_clusters` ran one thread per cell with ~60% of the lanes exiting immediately.
Added `compact_cluster_roots` (one thread per cell, writes the root cell index at the
cluster's slot) so that the aggregation runs with fully active warps, one thread per
cluster (launched over the cell count, since the cluster count is only known on the
device). All backends.
Result: `aggregate_clusters` 100.0 us vs 96.5 us before, plus the extra compaction kernel;
kernels/event 1.438 vs 1.428 ms. FAILED (no gain: the kernel is bound by the per-cluster
dependent-load chain, not by idle lanes). Reverted.

## Final validation (all 10 input events of ttbar_mu200, baseline build vs. final build)

A second build of the unmodified baseline (commit 4502b8795 + the dump helper) was made in a
git worktree and both builds dumped events 0-9:

* measurement sets and spacepoint sets: identical for all 10 events (order-independent
  comparison of surface id, local position, local variance, dimensions, diameter,
  subspace; spacepoints compared through their measurement content and global position);
* final build run twice: measurement and spacepoint lines bit-identical (0 differing
  lines over ~4.3M measurements and ~2.2M spacepoints), i.e. deterministic;
* track parameters: identical (bitwise) on every seed present in both runs / builds;
  the GBTS seed *sets* differ between any two runs, also for the unmodified baseline
  (GBTS uses atomics throughout; not modified per the task).
* CUDA unit tests: clustering, sorting, spacepoint formation, and all 640 CCA tests
  (`traccc-data-v10` downloaded into `Traccc/data`).

## Summary of the changes on the branch (commits on top of 4502b8795)

1. Deterministic radix-based measurement sorting (later reduced to one pass, then to a
   sortedness check + copy when already sorted).
2. Deterministic spacepoint formation (flags + prefix sum + scatter).
3. Fixed-size sorted measurement buffer (no host sync downstream).
4. Restructured CCL: parallel partition search, root flags + linked list output, prefix
   sum, separate aggregation kernel, deterministic slots.
5. Upload only the 4 cell columns used on the device; one-warp blocks for the parameter
   estimation kernel.
6. Default `target_cells_per_thread` 8 -> 4.
7. `traccc_dump_seeding_cuda` validation helper.

All algorithmic changes live in `device/common` and are wired into the CUDA, HIP, SYCL
and Alpaka backends (only CUDA could be compiled and tested in this checkout; the other
backends were edited to match and reviewed by hand, plus a host-side syntax check of the
common headers).

## Not done / ideas for further work

* GBTS seeding: per-event H2D copies of constant configuration (layer maps, LUTs) and
  its non-determinism (atomics) are the largest remaining items (~1.5 ms of the ~2.5 ms
  event), but the seeding algorithm was out of scope.
* The cell upload (4 x 4 MB per event, ~0.3 ms) is PCIe bound; a 16-bit channel EDM
  would halve it but touches the core EDM.
* `aggregate_clusters` (~97 us) and `ccl_kernel` (~100 us) are latency bound
  (dependent loads); an attempt to run the aggregation one thread per cluster did not
  help. Shared-memory caching of the cell coordinates in the CCL kernel was not tried.
* The throughput app processes events strictly sequentially on one stream (the TBB
  arena has one worker); `--threads=N` would overlap uploads/syncs of different events.

## Final benchmark (exact task command, 2000 events, 1 thread, load average ~25)

| build | ms/event | events/s |
|-------|----------|----------|
| baseline (4502b8795) | 7.853 | 127.3 |
| final (this branch), 3 runs | 2.370 / 2.374 / 2.360 | 422 / 421 / 424 |

Speed-up: 3.3x on the "seeding" throughput stage, of which ~1.5 ms/event is now the
(unchanged) GBTS seeding.
