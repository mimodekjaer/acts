/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

namespace detail {

/// Block-wide inclusive scan of one value per thread (Hillis-Steele in
/// shared memory, blockSize a power of two). Returns the inclusive prefix of
/// the calling thread; the block total is left in scratch[blockSize - 1]
/// until the next barrier-separated write to scratch.
template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_block_inclusive_scan(
    const barrier_t& barrier, vecmem::device_vector<unsigned int>& scratch,
    const unsigned int threadIndex, const unsigned int blockSize,
    const unsigned int value) {
  scratch[threadIndex] = value;
  barrier.blockBarrier();
  for (unsigned int offset = 1u; offset < blockSize; offset *= 2u) {
    const unsigned int add =
        (threadIndex >= offset) ? scratch[threadIndex - offset] : 0u;
    barrier.blockBarrier();
    scratch[threadIndex] += add;
    barrier.blockBarrier();
  }
  return scratch[threadIndex];
}

}  // namespace detail

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_build_edge_work_list(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_build_edge_work_list_payload& payload,
    const gbts_build_edge_work_list_shared_payload& shared_payload) {
  const vecmem::device_vector<const unsigned int> d_eta_node_counter(
      payload.eta_node_counter);
  const vecmem::device_vector<const uint2> d_bin_pairs(payload.bin_pairs);
  vecmem::device_vector<unsigned int> d_eta_bin_views(payload.eta_bin_views);
  vecmem::device_vector<unsigned int> d_pair_work_begin(
      payload.pair_work_begin);
  vecmem::device_vector<uint2> d_work_items(payload.work_items);
  vecmem::device_vector<unsigned int> scratch(shared_payload.scratch);

  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();

  // Every thread owns one contiguous strip of the input, scans it
  // sequentially, and one block scan of the strip totals gives the offsets:
  // one block scan per phase instead of one per blockSize elements.

  // 1. Node ranges of the eta bins.
  {
    const unsigned int strip = (payload.nEtaBins + blockSize - 1u) / blockSize;
    const unsigned int begin = threadIndex * strip;
    const unsigned int end =
        (begin + strip < payload.nEtaBins) ? begin + strip : payload.nEtaBins;
    unsigned int total = 0u;
    for (unsigned int bin = begin; bin < end; bin++) {
      total += d_eta_node_counter[bin];
    }
    const unsigned int inclusive = detail::gbts_block_inclusive_scan(
        barrier, scratch, threadIndex, blockSize, total);
    unsigned int running = inclusive - total;
    for (unsigned int bin = begin; bin < end; bin++) {
      const unsigned int count = d_eta_node_counter[bin];
      d_eta_bin_views[2u * bin] = running;
      running += count;
      d_eta_bin_views[2u * bin + 1u] = running;
    }
    if (threadIndex == 0u) {
      *payload.nNodes = scratch[blockSize - 1u];
    }
  }
  // The eta bin views are read by every thread below, and scratch is
  // rewritten.
  barrier.blockBarrier();

  // 2. Work items of the bin pairs.
  {
    const unsigned int strip =
        (payload.nBinPairs + blockSize - 1u) / blockSize;
    const unsigned int begin = threadIndex * strip;
    const unsigned int end = (begin + strip < payload.nBinPairs)
                                 ? begin + strip
                                 : payload.nBinPairs;
    unsigned int total = 0u;
    for (unsigned int pair = begin; pair < end; pair++) {
      const uint2 bins = d_bin_pairs[pair];
      const unsigned int n1 =
          d_eta_bin_views[2u * bins.x + 1u] - d_eta_bin_views[2u * bins.x];
      const unsigned int n2 =
          d_eta_bin_views[2u * bins.y + 1u] - d_eta_bin_views[2u * bins.y];
      if ((n1 > 0u) && (n2 > 0u)) {
        total += 1u + (n1 - 1u) / payload.chunkSize;
      }
    }
    const unsigned int inclusive = detail::gbts_block_inclusive_scan(
        barrier, scratch, threadIndex, blockSize, total);
    unsigned int running = inclusive - total;
    for (unsigned int pair = begin; pair < end; pair++) {
      const uint2 bins = d_bin_pairs[pair];
      const unsigned int n1 =
          d_eta_bin_views[2u * bins.x + 1u] - d_eta_bin_views[2u * bins.x];
      const unsigned int n2 =
          d_eta_bin_views[2u * bins.y + 1u] - d_eta_bin_views[2u * bins.y];
      d_pair_work_begin[pair] = running;
      if ((n1 > 0u) && (n2 > 0u)) {
        const unsigned int chunks = 1u + (n1 - 1u) / payload.chunkSize;
        for (unsigned int c = 0u; c < chunks; c++) {
          d_work_items[running + c] = uint2{pair, c};
        }
        running += chunks;
      }
    }
    if (threadIndex == 0u) {
      const unsigned int nWork = scratch[blockSize - 1u];
      d_pair_work_begin[payload.nBinPairs] = nWork;
      *payload.nWork = nWork;
    }
  }
}

}  // namespace traccc::device
