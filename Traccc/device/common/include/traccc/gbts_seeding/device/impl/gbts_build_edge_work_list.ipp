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

/// First index in [begin, end) of the sorted keys with key >= value
template <typename vector_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_key_lower_bound(
    const vector_t& keys, unsigned int begin, unsigned int end,
    const gbts_sort_key_t value) {
  while (begin < end) {
    const unsigned int mid = begin + (end - begin) / 2u;
    if (keys[mid] < value) {
      begin = mid + 1u;
    } else {
      end = mid;
    }
  }
  return begin;
}
template <typename vector_t>
TRACCC_HOST_DEVICE inline unsigned int gbts_key_lower_bound(
    const vector_t& keys, unsigned int end, const gbts_sort_key_t value) {
  return gbts_key_lower_bound(keys, 0u, end, value);
}

}  // namespace detail

template <concepts::thread_id1 thread_id_t, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void gbts_build_edge_work_list(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const gbts_build_edge_work_list_payload& payload,
    const gbts_build_edge_work_list_shared_payload& shared_payload) {
  const vecmem::device_vector<const gbts_sort_key_t> d_sort_keys(
      payload.sort_keys);
  const vecmem::device_vector<const uint2> d_bin_pairs(payload.bin_pairs);
  vecmem::device_vector<unsigned int> d_eta_bin_views(payload.eta_bin_views);
  vecmem::device_vector<unsigned int> d_pair_work_begin(
      payload.pair_work_begin);
  vecmem::device_vector<uint2> d_work_items(payload.work_items);
  vecmem::device_vector<unsigned int> d_bin_rads_bits(payload.bin_rads_bits);
  vecmem::device_vector<unsigned int> scratch(shared_payload.scratch);

  const unsigned int threadIndex = thread_id.getLocalThreadIdX();
  const unsigned int blockSize = thread_id.getBlockDimX();

  // Every thread owns one contiguous strip of the input, scans it
  // sequentially, and one block scan of the strip totals gives the offsets:
  // one block scan per phase instead of one per blockSize elements.

  // 1. Node ranges of the eta bins: the keys are sorted by (eta bin, phi)
  //    with the rejected keys last, so every bin's begin is a binary search
  //    and its end is the next bin's begin (the bins are contiguous).
  {
    const unsigned int nKeys = d_sort_keys.size();
    for (unsigned int bin = threadIndex; bin < payload.nEtaBins;
         bin += blockSize) {
      d_eta_bin_views[2u * bin] = detail::gbts_key_lower_bound(
          d_sort_keys, nKeys,
          static_cast<gbts_sort_key_t>(bin) << gbts_sort_key_eta_shift);
      // (min r, max r) accumulators of the bin (radii are >= 0, so their
      // float bits order like the values)
      d_bin_rads_bits[2u * bin] = gbts_float_bits(1e8f);
      d_bin_rads_bits[2u * bin + 1u] = gbts_float_bits(0.0f);
    }
    if (threadIndex == 0u) {
      *payload.nNodes = detail::gbts_key_lower_bound(
          d_sort_keys, nKeys,
          static_cast<gbts_sort_key_t>(payload.nEtaBins)
              << gbts_sort_key_eta_shift);
    }
    barrier.blockBarrier();
    const unsigned int nNodes = *payload.nNodes;
    for (unsigned int bin = threadIndex; bin < payload.nEtaBins;
         bin += blockSize) {
      d_eta_bin_views[2u * bin + 1u] = (bin + 1u < payload.nEtaBins)
                                           ? d_eta_bin_views[2u * bin + 2u]
                                           : nNodes;
    }
  }
  // The eta bin views are read by every thread below, and scratch is
  // rewritten.
  barrier.blockBarrier();

  // 2. Work items of the bin pairs.
  {
    const unsigned int strip = (payload.nBinPairs + blockSize - 1u) / blockSize;
    const unsigned int begin = threadIndex * strip;
    const unsigned int end =
        (begin + strip < payload.nBinPairs) ? begin + strip : payload.nBinPairs;
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
