// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

namespace ActsPlugins {

/// Lookup table from the surfaces of a detray detector to the module ids of
/// the module map, in device memory. Entry i is the module id of the surface
/// with index i in the detector, or kInvalid if it has none (portals, passive
/// surfaces, unmapped sensitive surfaces). This lets the module ids of space
/// points on the device be found directly from the surface links of their
/// measurements.
class DetrayModuleIdTable {
 public:
  static constexpr std::uint64_t kInvalid =
      std::numeric_limits<std::uint64_t>::max();

  /// @param detector detray detector, only its surface descriptors are used
  /// @param moduleId callable returning the module id of a detray surface
  ///        identifier, as std::optional<std::uint64_t>
  /// @param mr memory resource of the table (device memory)
  /// @param copy vecmem copy from host to @p mr
  template <typename detector_t, typename module_id_fn_t>
  DetrayModuleIdTable(const detector_t &detector,
                      const module_id_fn_t &moduleId,
                      vecmem::memory_resource &mr, vecmem::copy &copy) {
    const auto &surfaces = detector.surfaces();
    std::vector<std::uint64_t> table(surfaces.size(), kInvalid);
    for (const auto &sf : surfaces) {
      if (!sf.is_sensitive()) {
        continue;
      }
      ++m_nSensitive;
      if (const std::optional<std::uint64_t> id = moduleId(sf.identifier());
          id.has_value()) {
        table.at(sf.index()) = *id;
      } else {
        ++m_nUnmapped;
      }
    }
    if (m_nSensitive == m_nUnmapped) {
      throw std::invalid_argument(
          "DetrayModuleIdTable: no sensitive surface has a module id");
    }

    m_size = table.size();
    m_table = vecmem::make_unique_alloc<std::uint64_t[]>(mr, m_size);
    using view_t = vecmem::data::vector_view<std::uint64_t>;
    const auto size = static_cast<view_t::size_type>(m_size);
    copy(vecmem::data::vector_view<const std::uint64_t>(size, table.data()),
         view_t(size, m_table.get()), vecmem::copy::type::host_to_device)
        ->wait();
  }

  /// Device pointer to the table, indexed by the detray surface index
  const std::uint64_t *data() const { return m_table.get(); }
  /// Number of surfaces in the detector
  std::size_t size() const { return m_size; }
  /// Number of sensitive surfaces
  std::size_t nSensitive() const { return m_nSensitive; }
  /// Number of sensitive surfaces without a module id
  std::size_t nUnmapped() const { return m_nUnmapped; }

 private:
  vecmem::unique_alloc_ptr<std::uint64_t[]> m_table;
  std::size_t m_size = 0;
  std::size_t m_nSensitive = 0;
  std::size_t m_nUnmapped = 0;
};

}  // namespace ActsPlugins
