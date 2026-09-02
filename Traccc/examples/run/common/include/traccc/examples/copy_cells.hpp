/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/silicon_cell_collection.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::details {

/// Copy the cells needed by the clusterization to the device
///
/// The clusterization only uses the channel0, channel1, activation and
/// module_index columns of the cell collection; the time column is never
/// read on the device. Copying only the used columns saves ~20% of the
/// host-to-device transfer of the cells, which is PCIe bound.
///
/// @param copy   The (asynchronous) copy object to use
/// @param cells  The host cell collection
/// @param buffer The device buffer, already set up with the right capacity
/// @param wait   Wait for the copies to finish before returning
///
inline void copy_cells_for_clusterization(
    vecmem::copy& copy, const edm::silicon_cell_collection::host& cells,
    edm::silicon_cell_collection::buffer& buffer, bool wait = false) {
  const edm::silicon_cell_collection::const_data data = vecmem::get_data(cells);
  auto finish = [wait](vecmem::copy::event_type event) {
    if (wait) {
      event->wait();
    } else {
      event->ignore();
    }
  };
  finish(copy(data.get<0>(), buffer.get<0>()));  // channel0
  finish(copy(data.get<1>(), buffer.get<1>()));  // channel1
  finish(copy(data.get<2>(), buffer.get<2>()));  // activation
  finish(copy(data.get<4>(), buffer.get<4>()));  // module_index
}

}  // namespace traccc::details
