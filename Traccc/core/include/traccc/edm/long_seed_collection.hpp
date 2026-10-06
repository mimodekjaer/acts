// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

// Local include(s).
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc::edm {

/// Interface for seeds made of any number of spacepoints
///
/// Unlike @c traccc::edm::seed, which always holds a triplet, a long seed
/// keeps every spacepoint of a seed candidate, innermost first.
///
/// vecmem does not resize containers holding a jagged vector, so the number
/// of entries is fixed when the buffer is made, e.g. one entry per seed
/// candidate, each with room for the longest seed:
/// @code
/// long_seed_collection::buffer seeds(
///     std::vector<unsigned int>(n_candidates, max_spacepoints), mr, host_mr,
///     vecmem::data::buffer_type::resizable);
/// @endcode
/// After @c vecmem::copy::setup every entry is empty. A producer fills the
/// entry of its candidate with @c spacepoint_indices().push_back, and the
/// entries left empty hold no seed.
///
template <typename BASE>
class long_seed : public BASE {
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

  /// Indices of the spacepoints of the seed, innermost first (non-const)
  ///
  /// @return A (non-const) jagged vector of <tt>unsigned int</tt> values
  ///
  TRACCC_HOST_DEVICE
  auto& spacepoint_indices() { return BASE::template get<0>(); }
  /// Indices of the spacepoints of the seed, innermost first (const)
  ///
  /// @return A (const) jagged vector of <tt>unsigned int</tt> values
  ///
  TRACCC_HOST_DEVICE
  const auto& spacepoint_indices() const { return BASE::template get<0>(); }

  /// Quality of the seed (non-const)
  ///
  /// @return A (non-const) vector of <tt>float</tt> values
  ///
  TRACCC_HOST_DEVICE
  auto& quality() { return BASE::template get<1>(); }
  /// Quality of the seed (const)
  ///
  /// @return A (const) vector of <tt>float</tt> values
  ///
  TRACCC_HOST_DEVICE
  const auto& quality() const { return BASE::template get<1>(); }

  /// @}

};  // class long_seed

using long_seed_collection =
    vecmem::edm::container<long_seed,
                           vecmem::edm::type::jagged_vector<unsigned int>,
                           vecmem::edm::type::vector<float> >;

}  // namespace traccc::edm
