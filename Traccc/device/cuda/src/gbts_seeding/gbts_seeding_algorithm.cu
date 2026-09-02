/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_hits.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/device/gbts_build_edge_work_list.hpp"
#include "traccc/gbts_seeding/device/gbts_compress_graph.hpp"
#include "traccc/gbts_seeding/device/gbts_convert_seeds.hpp"
#include "traccc/gbts_seeding/device/gbts_count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_fill_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_make_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_match_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reindex_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/gbts_run_cca_iteration.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_bidding.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <algorithm>
#include <memory_resource>

// CUB include(s).
#include <cub/device/device_radix_sort.cuh>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

namespace traccc::cuda {

namespace kernels {

/// Widens a 1-byte flag to int (for the kept-flag prefix sum)
struct byte_to_int {
  __host__ __device__ int operator()(unsigned char k) const {
    return static_cast<int>(k);
  }
};

using float4 = traccc::float4;
using uint2 = traccc::uint2;
using int2 = traccc::int2;

// ---------------------------------------------------------------------------
// Stage 1 — nodes-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_bin_spacepoints
__global__ void gbts_bin_spacepoints(
    const device::gbts_bin_spacepoints_payload payload) {
  device::gbts_bin_spacepoints(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_build_edge_work_list
__global__ void gbts_build_edge_work_list(
    const device::gbts_build_edge_work_list_payload payload) {
  __shared__ unsigned int scratch[device::gbts_build_edge_work_list_block_size];
  const traccc::cuda::barrier barrier;
  device::gbts_build_edge_work_list(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<unsigned int>(
          device::gbts_build_edge_work_list_block_size, scratch)});
}

/// CUDA kernel for running @c traccc::device::gbts_sort_nodes
__global__ void gbts_sort_nodes(const device::gbts_sort_nodes_payload payload) {
  __shared__ unsigned int min_bits[device::gbts_sort_nodes_block_size];
  __shared__ unsigned int max_bits[device::gbts_sort_nodes_block_size];
  const traccc::cuda::barrier barrier;
  device::gbts_sort_nodes(details::thread_id1{}, barrier, payload,
                          {vecmem::data::vector_view<unsigned int>(
                               device::gbts_sort_nodes_block_size, min_bits),
                           vecmem::data::vector_view<unsigned int>(
                               device::gbts_sort_nodes_block_size, max_bits)});
}

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_make_graph_edges
///
/// @tparam fill false: count the edges, true: write them
template <bool fill>
__global__ void gbts_make_graph_edges(
    const device::gbts_make_graph_edges_payload payload) {
  __shared__ unsigned int
      work_slot[traccc::device::gbts_make_graph_edges_scratch_size];
  const traccc::cuda::barrier barrier;

  device::gbts_make_graph_edges<fill>(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<unsigned int>(
          traccc::device::gbts_make_graph_edges_scratch_size, work_slot)});
}

/// CUDA kernel for running @c traccc::device::gbts_match_graph_edges
__global__ void gbts_match_graph_edges(
    const device::gbts_match_graph_edges_payload payload) {
  device::gbts_match_graph_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_compress_graph
__global__ void gbts_compress_graph(
    const device::gbts_compress_graph_payload payload) {
  device::gbts_compress_graph(details::thread_id1{}, payload);
}

// ---------------------------------------------------------------------------
// Stage 3 — graph-processing kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_run_cca_iteration
__global__ void gbts_run_cca_iteration(
    const device::gbts_run_cca_iteration_payload payload) {
  __shared__ unsigned int changed[1];
  const traccc::cuda::barrier barrier;
  device::gbts_run_cca_iteration(
      details::thread_id1{}, barrier, payload,
      {vecmem::data::vector_view<unsigned int>(1u, changed)});
}

/// CUDA kernel for running @c traccc::device::gbts_count_terminus_edges
__global__ void gbts_count_terminus_edges(
    const device::gbts_count_terminus_edges_payload payload) {
  device::gbts_count_terminus_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_fill_path_store
__global__ void gbts_fill_path_store(
    const device::gbts_fill_path_store_payload payload) {
  device::gbts_fill_path_store(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_reset_edge_bids
__global__ void gbts_reset_edge_bids(
    const device::gbts_reset_edge_bids_payload payload) {
  device::gbts_reset_edge_bids(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_rebid_seeds_for_edges
__global__ void gbts_rebid_seeds_for_edges(
    const device::gbts_rebid_seeds_for_edges_payload payload) {
  device::gbts_rebid_seeds_for_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bid_seeds_for_hits
__global__ void gbts_bid_seeds_for_hits(
    const device::gbts_bid_seeds_for_hits_payload payload) {
  device::gbts_bid_seeds_for_hits(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_convert_seeds
__global__ void gbts_convert_seeds(
    const device::gbts_convert_seeds_payload payload) {
  device::gbts_convert_seeds(details::thread_id1{}, payload);
}

}  // namespace kernels

// ===========================================================================
// gbts_seeding_algorithm: kernel launchers
// ===========================================================================

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    const vecmem::copy& copy, const stream_wrapper& str,
    std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      cuda::algorithm_base{str} {}

void gbts_seeding_algorithm::gbts_bin_spacepoints_kernel(
    const device::gbts_bin_spacepoints_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
  kernels::gbts_bin_spacepoints<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_sort_node_keys_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  // Keys-only radix sort of the (eta bin, quantised phi) bits above the
  // spacepoint index: 3 one-sweep passes at most, stable in the index.
  unsigned int eta_bits = 0u;
  while ((1u << eta_bits) <= payload.nEtaBins) {
    ++eta_bits;
  }
  const int begin_bit = static_cast<int>(device::gbts_sort_key_phi_shift);
  const int end_bit =
      static_cast<int>(device::gbts_sort_key_eta_shift + eta_bits);

  cudaStream_t cuda_stream = details::get_stream(stream());
  vecmem::data::vector_buffer<device::gbts_sort_key_t> keys_alt(payload.nKeys,
                                                                mr().main);
  cub::DoubleBuffer<device::gbts_sort_key_t> d_keys(payload.sort_keys.ptr(),
                                                    keys_alt.ptr());
  std::size_t temp_bytes = 0u;
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortKeys(
      nullptr, temp_bytes, d_keys, static_cast<int>(payload.nKeys), begin_bit,
      end_bit, cuda_stream));
  vecmem::data::vector_buffer<char> temp(
      static_cast<unsigned int>(std::max<std::size_t>(temp_bytes, 1u)),
      mr().main);
  TRACCC_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortKeys(
      temp.ptr(), temp_bytes, d_keys, static_cast<int>(payload.nKeys),
      begin_bit, end_bit, cuda_stream));
  if (d_keys.Current() != payload.sort_keys.ptr()) {
    // The sorted keys ended up in the alternate buffer.
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(payload.sort_keys.ptr(), d_keys.Current(),
                        payload.nKeys * sizeof(device::gbts_sort_key_t),
                        cudaMemcpyDeviceToDevice, cuda_stream));
  }
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {
  const unsigned int n_threads = device::gbts_sort_nodes_block_size;
  const unsigned int n_blocks = 1 + (payload.nKeys - 1) / n_threads;
  kernels::gbts_sort_nodes<<<n_blocks, n_threads, 0,
                             details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_build_edge_work_list_kernel(
    const device::gbts_build_edge_work_list_payload& payload) const {
  kernels::gbts_build_edge_work_list<<<
      1, device::gbts_build_edge_work_list_block_size, 0,
      details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_count_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  // One thread per inner node of a chunk: the block size must equal the
  // chunk size.
  const unsigned int n_threads =
      traccc::device::gbts_consts::node_buffer_length;
  // The blocks stride over the device-side work list.
  const unsigned int n_blocks =
      std::min(payload.nWorkMax, device::gbts_make_graph_edges_max_blocks);
  kernels::gbts_make_graph_edges<false>
      <<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  // Turn the per-node edge counts into the bucket begin/end offsets.
  vecmem::device_vector<unsigned int> d_num_outgoing_edges(
      payload.num_outgoing_edges);
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      d_num_outgoing_edges.begin(), d_num_outgoing_edges.end(),
      d_num_outgoing_edges.begin());
}

void gbts_seeding_algorithm::gbts_make_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {
  const unsigned int n_threads =
      traccc::device::gbts_consts::node_buffer_length;
  // One block per work item (static replay), the blocks then stride over
  // the overflow list.
  const unsigned int n_blocks =
      std::min(payload.nWorkMax, device::gbts_make_graph_edges_max_fill_blocks);
  kernels::gbts_make_graph_edges<true>
      <<<n_blocks, n_threads, 0, details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_match_graph_edges_kernel(
    const device::gbts_match_graph_edges_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
  kernels::gbts_match_graph_edges<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {
  // The 1-byte flags are widened on the fly; the sum is accumulated in int.
  const unsigned char* kept = payload.kept.ptr();
  auto kept_int = thrust::make_transform_iterator(kept, kernels::byte_to_int{});
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      kept_int, kept_int + payload.nEdgesMax, payload.reIndexer.ptr());
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks =
      std::min(1u + (payload.nEdgesMax - 1u) / n_threads, 8192u);
  kernels::gbts_compress_graph<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_run_cca_iteration_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {
  const unsigned int n_threads = 256;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;

  kernels::gbts_run_cca_iteration<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_count_terminus_edges_kernel(
    const device::gbts_count_terminus_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
  kernels::gbts_count_terminus_edges<<<n_blocks, n_threads, 0,
                                       details::get_stream(stream())>>>(
      payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
  // Lay out the path store: each terminus edge gets a contiguous row range.
  vecmem::device_vector<unsigned int> d_row_sizes(payload.row_sizes);
  thrust::inclusive_scan(
      thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
          .on(details::get_stream(stream())),
      d_row_sizes.begin(), d_row_sizes.begin() + payload.nConnectedEdges,
      d_row_sizes.begin());
  // Publish the total row count (last entry of the scanned row sizes).
  TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
      payload.row_count, d_row_sizes.data() + payload.nConnectedEdges - 1u,
      sizeof(unsigned int), cudaMemcpyDeviceToDevice,
      details::get_stream(stream())));
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_fill_path_store<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_reset_edge_bids<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_rebid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                        details::get_stream(stream())>>>(
      payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_bid_seeds_for_hits<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {
  const unsigned int n_threads = 128;
  const unsigned int n_blocks = 1 + (payload.nRowsGrid - 1) / n_threads;
  kernels::gbts_convert_seeds<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(payload);
  TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
