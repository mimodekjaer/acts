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
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/stream_wrapper.hpp>

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

/// Device memory and copies for the work of one stage on one CUDA stream.
///
/// Allocations come from the memory resource of the execution context, or, if
/// it has none, from a stream-ordered resource on the stream of the context.
/// A resource from the context is used the same way: memory may be released
/// while work on the stream that uses it is still pending, so it must only be
/// handed out again to work ordered after that (e.g. a stream-ordered or a
/// per-stream caching resource).
///
/// Copies and memsets are vecmem operations on the stream, in stream order
/// like the kernels. Copies to pageable host memory complete when they return.
class DeviceMemory {
 public:
  DeviceMemory(cudaStream_t stream, vecmem::memory_resource *mr)
      : m_stream(stream), m_streamWrapper(stream), m_copy(m_streamWrapper) {
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

  /// vecmem copy object of the stream
  vecmem::cuda::async_copy &copy() { return m_copy; }

  /// Copy @p n elements from host to device memory
  template <typename T>
  void toDevice(T *device, const T *host, std::size_t n) {
    copyImpl(device, host, n, vecmem::copy::type::host_to_device);
  }

  /// Copy @p n elements from device to host memory
  template <typename T>
  void toHost(T *host, const T *device, std::size_t n) {
    copyImpl(host, device, n, vecmem::copy::type::device_to_host);
  }

  /// Copy @p n elements between two device arrays
  template <typename T>
  void copyDevice(T *to, const T *from, std::size_t n) {
    copyImpl(to, from, n, vecmem::copy::type::device_to_device);
  }

  /// Set the bytes of @p n device elements to @p value
  template <typename T>
  void memset(T *device, std::size_t n, int value) {
    if (n == 0) {
      return;
    }
    // As bytes: vecmem views do not support every element type (e.g. bool)
    m_copy
        .memset(view(reinterpret_cast<std::byte *>(device), n * sizeof(T)),
                value)
        ->ignore();
  }

  /// Wait for all work on the stream
  void synchronize() { m_streamWrapper.synchronize(); }

 private:
  template <typename T>
  static vecmem::data::vector_view<T> view(T *ptr, std::size_t n) {
    using size_type = typename vecmem::data::vector_view<T>::size_type;
    return {static_cast<size_type>(n), ptr};
  }

  template <typename T>
  void copyImpl(T *to, const T *from, std::size_t n,
                vecmem::copy::type::copy_type type) {
    if (n == 0) {
      return;
    }
    m_copy(view(from, n), view(to, n), type)->ignore();
  }

  cudaStream_t m_stream{};
  vecmem::cuda::stream_wrapper m_streamWrapper;
  vecmem::cuda::async_copy m_copy;
  std::optional<CudaStreamOrderedMemoryResource> m_ownResource;
  vecmem::memory_resource *m_mr = nullptr;
  std::optional<ThrustAllocator> m_thrustAllocator;
};

}  // namespace ActsPlugins::detail
