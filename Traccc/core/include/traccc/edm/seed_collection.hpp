// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

// System include(s)
#include <array>
#include <ostream>

namespace traccc::edm {

/// The largest number of spacepoints a seed can hold
inline constexpr unsigned int seed_max_spacepoints = 16u;

/// Interface for the @c traccc::edm::seed_collection class.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays of the SoA containers, or the variables of the AoS proxies
/// created on top of the SoA containers.
///
template <typename BASE>
class seed : public BASE {
 public:
  /// @name Functions inherited from the base class
  /// @{

  /// Inherit the base class's constructor(s)
  using BASE::BASE;
  /// Inherit the base class's assignment operator(s).
  using BASE::operator=;

  /// @}

  /// @name Seed Information
  /// @{

  /// Indices of the spacepoints, innermost first (non-const)
  ///
  /// Only the first @c n_spacepoints() entries of a seed are used.
  ///
  /// @return A (non-const) vector of
  ///         <tt>std::array<unsigned int, seed_max_spacepoints></tt> values
  ///
  TRACCC_HOST_DEVICE
  auto& spacepoint_indices() { return BASE::template get<0>(); }
  /// Indices of the spacepoints, innermost first (const)
  ///
  /// Only the first @c n_spacepoints() entries of a seed are used.
  ///
  /// @return A (const) vector of
  ///         <tt>std::array<unsigned int, seed_max_spacepoints></tt> values
  ///
  TRACCC_HOST_DEVICE
  const auto& spacepoint_indices() const { return BASE::template get<0>(); }

  /// Number of spacepoints of the seed (non-const)
  ///
  /// @return A (non-const) vector of <tt>unsigned int</tt> values
  ///
  TRACCC_HOST_DEVICE
  auto& n_spacepoints() { return BASE::template get<1>(); }
  /// Number of spacepoints of the seed (const)
  ///
  /// @return A (const) vector of <tt>unsigned int</tt> values
  ///
  TRACCC_HOST_DEVICE
  const auto& n_spacepoints() const { return BASE::template get<1>(); }

  /// Quality of the seed (const)
  ///
  /// @return A (const) vector of <tt>float</tt> values
  ///
  TRACCC_HOST_DEVICE
  const auto& quality() const { return BASE::template get<2>(); }
  /// Quality of the seed (non-const)
  ///
  /// @return A (non-const) vector of <tt>float</tt> values
  ///
  TRACCC_HOST_DEVICE
  auto& quality() { return BASE::template get<2>(); }

  /// Index of the bottom (innermost) spacepoint
  ///
  /// @note This function must only be used on proxy objects, not on
  ///       containers!
  ///
  TRACCC_HOST_DEVICE
  unsigned int bottom_index() const { return spacepoint_indices()[0]; }

  /// Index of the middle spacepoint, the one at @c n_spacepoints()/2
  ///
  /// @note This function must only be used on proxy objects, not on
  ///       containers!
  ///
  TRACCC_HOST_DEVICE
  unsigned int middle_index() const {
    return spacepoint_indices()[n_spacepoints() / 2u];
  }

  /// Index of the top (outermost) spacepoint
  ///
  /// @note This function must only be used on proxy objects, not on
  ///       containers!
  ///
  TRACCC_HOST_DEVICE
  unsigned int top_index() const {
    return spacepoint_indices()[n_spacepoints() - 1u];
  }

  /// @}

  /// @name Utility functions
  /// @{

  /// Equality operator
  ///
  /// @note This function must only be used on proxy objects, not on
  ///       containers!
  ///
  /// @param[in] other The object to compare with
  /// @return @c true if the objects are equal, @c false otherwise
  ///
  template <typename T>
  TRACCC_HOST_DEVICE bool operator==(const seed<T>& other) const;

  /// Comparison operator
  ///
  /// @note This function must only be used on proxy objects, not on
  ///       containers!
  ///
  /// @param[in] other The object to compare with
  /// @return A strong ordering object, describing the relation between the
  ///         two objects
  ///
  template <typename T>
  TRACCC_HOST_DEVICE std::strong_ordering operator<=>(
      const seed<T>& other) const;

  /// @}

 private:
  /// @returns a string stream that prints the seed details
  TRACCC_HOST
  friend std::ostream& operator<<(std::ostream& os, const seed& s) {
    os << "quality: " << s.quality() << std::endl;
    os << "measurements: [bottom = " << s.bottom_index()
       << ", middle = " << s.middle_index() << ", top = " << s.top_index()
       << "]" << std::endl;

    return os;
  }

};  // class seed

/// SoA container describing reconstructed track seeds
using seed_collection = vecmem::edm::container<
    seed,
    vecmem::edm::type::vector<std::array<unsigned int, seed_max_spacepoints> >,
    vecmem::edm::type::vector<unsigned int>, vecmem::edm::type::vector<float> >;

}  // namespace traccc::edm

// Include the implementation.
#include "traccc/edm/impl/seed_collection.ipp"
