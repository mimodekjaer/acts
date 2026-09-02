/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <mutex>

#include <vecmem/memory/device_address_space.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/device/mutex.hpp"
#include "traccc/device/unique_lock.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"

namespace traccc::device {

/// Implementation of a FastSV algorithm with the following steps:
///   1) mix of stochastic and aggressive hooking
///   2) shortcutting
///
/// The implementation corresponds to an adapted versiion of Algorithm 3 of
/// the following paper:
/// https://www.sciencedirect.com/science/article/pii/S0743731520302689
///
///                     This array only gets updated at the end of the iteration
///                     to prevent race conditions.
/// @param[in] adjc     The number of adjacent cells
/// @param[in] adjv     Vector of adjacent cells
/// @param[in] tid      The thread index
/// @param[in] blckDim  The block size
/// @param[inout] f     array holding the parent cell ID for the current
///                     iteration.
/// @param[inout] gf    array holding grandparent cell ID from the previous
///                     iteration.
/// @param[in] barrier  A generic object for block-wide synchronisation
///
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t, typename index_t>
TRACCC_HOST_DEVICE void fast_sv_1(const thread_id_t& thread_id,
                                  vecmem::device_vector<index_t>& f,
                                  vecmem::device_vector<index_t>& gf,
                                  unsigned char* adjc, index_t* adjv,
                                  unsigned int thread_cell_count,
                                  barrier_t& barrier) {
  /*
   * The algorithm finishes if an iteration leaves the arrays unchanged.
   * This varible will be set if a change is made, and dictates if another
   * loop is necessary.
   */
  bool gf_changed;

  do {
    /*
     * Reset the end-parameter to false, so we can set it to true if we
     * make a change to the gf array.
     */
    gf_changed = false;

    /*
     * The algorithm executes in a loop of three distinct parallel
     * stages. In this first one, a mix of stochastic and aggressive
     * hooking, we examine adjacent cells and copy their grand parents
     * cluster ID if it is lower than ours, essentially merging the two
     * together.
     */
    for (unsigned int tst = 0; tst < thread_cell_count; ++tst) {
      const unsigned int cid =
          tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();

      TRACCC_ASSUME(adjc[tst] <= 4);
      for (unsigned char k = 0; k < adjc[tst]; ++k) {
        const auto cid2 = adjv[4 * tst + k];

        index_t q2 = gf.at(cid2);
        index_t q1 = gf.at(cid);

        if (gf.at(cid) > q2) {
          f.at(f.at(cid)) = q2;
          f.at(cid) = q2;
        }
        if (gf.at(cid2) > q1) {
          f.at(f.at(cid2)) = q1;
          f.at(cid2) = q1;
        }
      }
    }

    /*
     * Each stage in this algorithm must be preceded by a
     * synchronization barrier!
     */
    barrier.blockBarrier();

    for (unsigned int tst = 0; tst < thread_cell_count; ++tst) {
      const unsigned int cid =
          tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
      /*
       * The second stage is shortcutting, which is an optimisation that
       * allows us to look at any shortcuts in the cluster IDs that we
       * can merge without adjacency information.
       */
      if (f.at(cid) > gf.at(cid)) {
        f.at(cid) = gf.at(cid);
      }
    }

    /*
     * Synchronize before the final stage.
     */
    barrier.blockBarrier();

    for (unsigned int tst = 0; tst < thread_cell_count; ++tst) {
      const unsigned int cid =
          tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
      /*
       * Update the array for the next generation, keeping track of any
       * changes we make.
       */
      if (gf.at(cid) != f.at(f.at(cid))) {
        gf.at(cid) = f.at(f.at(cid));
        gf_changed = true;
      }
    }

    /*
     * To determine whether we need another iteration, we use block
     * voting mechanics. Each thread checks if it has made any changes
     * to the arrays, and votes. If any thread votes true, all threads
     * will return a true value and go to the next iteration. Only if
     * all threads return false will the loop exit.
     */
  } while (barrier.blockOr(gf_changed));
}

template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t, typename index_t>
TRACCC_HOST_DEVICE inline void ccl_core(
    const thread_id_t& thread_id, const unsigned int partition_start,
    const unsigned int partition_end, vecmem::device_vector<index_t> f,
    vecmem::device_vector<index_t> gf, index_t* adjv, unsigned char* adjc,
    const edm::silicon_cell_collection::const_device& cells_device,
    const barrier_t& barrier, vecmem::device_vector<unsigned int>& cluster_flags,
    vecmem::device_vector<unsigned int>& next_cell) {
  const unsigned int size = partition_end - partition_start;

  assert(size <= f.size());
  assert(size <= gf.size());

  const unsigned int thread_cell_count =
      (size - thread_id.getLocalThreadIdX() + thread_id.getBlockDimX() - 1) /
      thread_id.getBlockDimX();

  for (unsigned int tst = 0; tst < thread_cell_count; ++tst) {
    /*
     * Look for adjacent cells to the current one.
     */
    const unsigned int cid =
        tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
    adjc[tst] = 0;

    reduce_problem_cell(cells_device, cid, partition_start, partition_end,
                        adjc[tst], &adjv[4 * tst]);

    f.at(cid) = static_cast<index_t>(cid);
    gf.at(cid) = static_cast<index_t>(cid);
  }

  /*
   * Now that the data has initialized, we synchronize again before we
   * move onto the actual processing part.
   */
  barrier.blockBarrier();

  /*
   * Run FastSV algorithm, which will update the father index to that of
   * the cell belonging to the same cluster with the lowest index.
   */
  fast_sv_1(thread_id, f, gf, adjc, adjv, thread_cell_count, barrier);

  barrier.blockBarrier();

  /*
   * We'll now convert the parent array `f` into a linked list equivalent
   * stored in array `gf`. The point of this linked list is that the ID of
   * each cell is either:
   *
   * - An out-of-bounds value if there is no later cell in the same
   *   cluster; or
   * - An in-bounds index pointing to the next cell belonging to the same
   *   cluster.
   *
   * If, for example, f would look like this:
   *
   *        0  1  2  3  4  5  6  7  8
   * f  = [ 0, 0, 2, 0, 2, 5, 2, 5, 8]
   *
   * We would compute:
   *
   * gf = [ 1, 3, 4, 9, 6, 7, 9, 9, 9]
   *
   * This makes aggregating the clusters trivial.
   *
   * WARNING: After this point, the `gf` vector no longer contains the
   * grandparent information it held before. Do not use it as such!
   *
   * First, we start out by setting out-of-bounds values for all cells...
   */
  for (unsigned int i = thread_id.getLocalThreadIdX(); i < size;
       i += thread_id.getBlockDimX()) {
    gf.at(i) = static_cast<index_t>(size);
  }

  barrier.blockBarrier();

  /*
   * Now we construct the actual linked list. We move backwards here,
   * because we are much more likely to be able to exit our loop early
   * compared to looping forwards.
   */
  for (unsigned int i = thread_id.getLocalThreadIdX(); i < size;
       i += thread_id.getBlockDimX()) {
    const index_t effi = static_cast<index_t>(size - (i + 1));

    const auto fid = f.at(effi);

    if (fid != effi) {
      for (unsigned int j = effi - 1; j < size; --j) {
        if (fid == f.at(j)) {
          gf.at(j) = effi;
          break;
        }
      }
    }
  }

  barrier.blockBarrier();

  /*
   * Finally, write out the root flags and the linked list in global
   * (cell) indices. The measurements are created by a separate kernel
   * (@c aggregate_clusters), once the root flags have been turned into
   * prefix sums, which gives every measurement a deterministic position.
   */
  for (unsigned int i = thread_id.getLocalThreadIdX(); i < size;
       i += thread_id.getBlockDimX()) {
    const unsigned int pos = partition_start + i;
    cluster_flags.at(pos) = (f.at(i) == i) ? 1u : 0u;
    const unsigned int next = gf.at(i);
    next_cell.at(pos) =
        (next < size) ? (partition_start + next) : details::INVALID_CELL;
  }
}

/// Check whether a partition may start at a given cell
///
/// Partitions may only be split at module boundaries, or at gaps in the
/// channel1 coordinate: a cell can never be adjacent to a cell in front of
/// such a position.
///
TRACCC_HOST_DEVICE inline bool is_partition_split(
    const edm::silicon_cell_collection::const_device& cells,
    const unsigned int i) {
  if (i == 0u || i >= cells.size()) {
    return true;
  }
  return (cells.module_index().at(i - 1) != cells.module_index().at(i)) ||
         (cells.channel1().at(i) > cells.channel1().at(i - 1) + 1);
}

/// Find the first valid partition split point at or after a given position
///
/// Parallel search: every thread of the block tests one candidate position
/// per iteration, and the minimum valid position is collected with an atomic
/// operation on the shared memory variable. The result is exactly the same as
/// that of a sequential walk from @c from.
///
/// @param[in]    thread_id The thread identifier object
/// @param[in]    barrier   The block barrier
/// @param[in]    cells     The cell collection
/// @param[in]    from      The position to start the search from
/// @param[inout] result    Shared memory variable, initialised to
///                         @c cells.size() by the caller, receiving the
///                         result
///
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void find_partition_split(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const edm::silicon_cell_collection::const_device& cells,
    const unsigned int from, unsigned int& result) {
  vecmem::device_atomic_ref<unsigned int, vecmem::device_address_space::local>
      atomic_result(result);
  unsigned int i = from + thread_id.getLocalThreadIdX();
  bool valid = false;
  do {
    valid = is_partition_split(cells, i);
    if (valid) {
      atomic_result.fetch_min(i);
    }
    i += thread_id.getBlockDimX();
  } while (!barrier.blockOr(valid));
}

template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void ccl_kernel(
    const clustering_config cfg, const thread_id_t& thread_id,
    const edm::silicon_cell_collection::const_view& cells_view,
    unsigned int& partition_start, unsigned int& partition_end,
    vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view,
    vecmem::data::vector_view<details::fallback_index_t> f_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> adjv_backup_view,
    vecmem::device_atomic_ref<uint32_t> backup_mutex, const barrier_t& barrier,
    vecmem::data::vector_view<unsigned int> cluster_flags_view,
    vecmem::data::vector_view<unsigned int> next_cell_view) {
  // Construct device containers around the views.
  const edm::silicon_cell_collection::const_device cells_device(cells_view);
  vecmem::device_vector<details::index_t> f_primary(f_view);
  vecmem::device_vector<details::index_t> gf_primary(gf_view);
  vecmem::device_vector<details::fallback_index_t> f_backup(f_backup_view);
  vecmem::device_vector<details::fallback_index_t> gf_backup(gf_backup_view);
  vecmem::device_vector<unsigned char> adjc_backup(adjc_backup_view);
  vecmem::device_vector<details::fallback_index_t> adjv_backup(
      adjv_backup_view);
  vecmem::device_vector<unsigned int> cluster_flags(cluster_flags_view);
  vecmem::device_vector<unsigned int> next_cell(next_cell_view);

  mutex<uint32_t> mutex(backup_mutex);
  unique_lock lock(mutex, std::defer_lock);

  const unsigned int num_cells = cells_device.size();

  /*
   * First, we determine the exact range of cells that is to be examined
   * by this block of threads. We start from an initial range determined
   * by the block index multiplied by the target number of cells per
   * block. We then shift both the start and the end of the block forward
   * (to a later point in the array) to the first positions where a
   * partition may be split. Start and end may be moved different amounts.
   *
   * The end of this partition is the start of the next one, so the
   * partitions tile the cell collection exactly.
   */
  const unsigned int naive_start =
      thread_id.getBlockIdX() * cfg.target_partition_size();
  assert(naive_start < num_cells);
  const unsigned int naive_end =
      std::min(num_cells, naive_start + cfg.target_partition_size());

  if (thread_id.getLocalThreadIdX() == 0) {
    partition_start = num_cells;
    partition_end = num_cells;
  }

  barrier.blockBarrier();

  find_partition_split(thread_id, barrier, cells_device, naive_start,
                       partition_start);
  find_partition_split(thread_id, barrier, cells_device, naive_end,
                       partition_end);

  barrier.blockBarrier();

  assert(partition_start <= partition_end);

  // It seems that sycl runs into undefined behaviour when calling
  // group synchronisation functions when some threads have already run
  // into a return. As such, we cannot use returns in this kernel.

  // Get partition for this thread group
  const unsigned int size = partition_end - partition_start;

  // If the size is zero, we can just retire the whole block.
  if (size == 0) {
    return;
  }

  /*
   * If our partition is too large, we need to handle this specific edge
   * case. The first thread of the block will attempt to enter a critical
   * section by obtaining a lock on a mutex in global memory. When this is
   * obtained, we can use some memory in global memory instead of the shared
   * memory. This can be done more efficiently, but this should be a very
   * rare edge case.
   */
  if (size > cfg.max_partition_size()) [[unlikely]] {
    if (thread_id.getLocalThreadIdX() == 0) {
      lock.lock();
    }

    barrier.blockBarrier();

    unsigned char* adjc = adjc_backup.data() + (thread_id.getLocalThreadIdX() *
                                                cfg.max_cells_per_thread *
                                                cfg.backup_size_multiplier);
    details::fallback_index_t* adjv =
        adjv_backup.data() +
        (thread_id.getLocalThreadIdX() * 4 * cfg.max_cells_per_thread *
         cfg.backup_size_multiplier);
    ccl_core(thread_id, partition_start, partition_end, f_backup, gf_backup,
             adjv, adjc, cells_device, barrier, cluster_flags, next_cell);
  } else {
    /*
     * Vector of indices of the adjacent cells.
     */
    details::index_t adjv[details::CELLS_PER_THREAD_STACK_LIMIT * 4];

    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    unsigned char adjc[details::CELLS_PER_THREAD_STACK_LIMIT];

    ccl_core(thread_id, partition_start, partition_end, f_primary, gf_primary,
             adjv, adjc, cells_device, barrier, cluster_flags, next_cell);
  }

  barrier.blockBarrier();
}
}  // namespace traccc::device
