// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsPlugins/Gnn/Tensor.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

#include <cstddef>
#include <optional>

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace ActsPlugins::detail {

/// vecmem memory resource for stream-ordered device memory
/// (cudaMallocAsync / cudaFreeAsync on one stream). Memory released while work
/// on the stream is still pending is only reused by work ordered after it, so
/// no synchronization is needed when freeing.
class CudaStreamOrderedMemoryResource final : public vecmem::memory_resource {
 public:
  explicit CudaStreamOrderedMemoryResource(cudaStream_t stream)
      : m_stream(stream) {}

  cudaStream_t stream() const { return m_stream; }

 private:
  void *do_allocate(std::size_t bytes, std::size_t /*alignment*/) override {
    // The stream-ordered allocator aligns to at least 256 bytes
    void *ptr = nullptr;
    if (bytes > 0) {
      ACTS_CUDA_CHECK(cudaMallocAsync(&ptr, bytes, m_stream));
    }
    return ptr;
  }

  void do_deallocate(void *ptr, std::size_t /*bytes*/,
                     std::size_t /*alignment*/) override {
    // Called from destructors, so this must not throw. A failure here leaves
    // a sticky CUDA error that is reported by the next checked CUDA call.
    if (ptr != nullptr) {
      static_cast<void>(cudaFreeAsync(ptr, m_stream));
    }
  }

  bool do_is_equal(
      const vecmem::memory_resource &other) const noexcept override {
    const auto *o =
        dynamic_cast<const CudaStreamOrderedMemoryResource *>(&other);
    return o != nullptr && o->m_stream == m_stream;
  }

  cudaStream_t m_stream;
};

/// Byte allocator for thrust execution policies on top of a vecmem resource
class ThrustAllocator {
 public:
  using value_type = char;

  explicit ThrustAllocator(vecmem::memory_resource &mr) : m_mr(&mr) {}

  char *allocate(std::size_t n) {
    return static_cast<char *>(m_mr->allocate(n));
  }
  void deallocate(char *ptr, std::size_t n) { m_mr->deallocate(ptr, n); }

 private:
  vecmem::memory_resource *m_mr;
};

/// Device memory for the work of one stage on one CUDA stream.
///
/// Allocations come from the memory resource of the execution context, or, if
/// it has none, from a stream-ordered resource on the stream of the context.
/// A resource from the context is used the same way: memory may be released
/// while work on the stream that uses it is still pending, so it must only be
/// handed out again to work ordered after that (e.g. a stream-ordered or a
/// per-stream caching resource).
class DeviceMemory {
 public:
  DeviceMemory(cudaStream_t stream, vecmem::memory_resource *mr)
      : m_stream(stream) {
    if (mr == nullptr) {
      mr = &m_ownResource.emplace(stream);
    }
    m_mr = mr;
    m_thrustAllocator.emplace(*m_mr);
  }

  explicit DeviceMemory(const ExecutionContext &ctx)
      : DeviceMemory(ctx.stream.value(), ctx.memoryResource) {}

  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  DeviceMemory(DeviceMemory &&) = delete;
  DeviceMemory &operator=(DeviceMemory &&) = delete;

  cudaStream_t stream() const { return m_stream; }
  vecmem::memory_resource &resource() const { return *m_mr; }

  /// Uninitialized device array of @p n elements
  template <typename T>
  vecmem::unique_alloc_ptr<T[]> make(std::size_t n) const {
    return vecmem::make_unique_alloc<T[]>(*m_mr, n);
  }

  /// Thrust execution policy on the stream that does not synchronize and
  /// takes its temporary memory from resource()
  auto policy() {
    return thrust::cuda::par_nosync(*m_thrustAllocator).on(m_stream);
  }

 private:
  cudaStream_t m_stream{};
  std::optional<CudaStreamOrderedMemoryResource> m_ownResource;
  vecmem::memory_resource *m_mr = nullptr;
  std::optional<ThrustAllocator> m_thrustAllocator;
};

}  // namespace ActsPlugins::detail
