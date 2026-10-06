// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Project include(s).
#include "traccc/edm/long_seed_collection.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Google test include(s).
#include <gtest/gtest.h>

// System include(s).
#include <vector>

TEST(LongSeedCollection, FillAndCopy) {
  vecmem::host_memory_resource host_mr;
  vecmem::copy copy;

  // One entry per seed candidate, the middle one left empty.
  const std::vector<std::vector<unsigned int>> input = {
      {4, 8, 15, 16}, {}, {23, 42, 7, 9, 11}};
  const std::vector<float> qualities = {1.5f, 0.f, 2.5f};

  traccc::edm::long_seed_collection::buffer buffer(
      std::vector<unsigned int>(input.size(), 16u), host_mr, nullptr,
      vecmem::data::buffer_type::resizable);
  copy.setup(buffer)->wait();

  // Fill it the way a kernel would.
  {
    traccc::edm::long_seed_collection::device seeds(buffer);
    for (unsigned int i = 0; i < input.size(); ++i) {
      auto seed = seeds.at(i);
      for (const unsigned int sp : input[i]) {
        seed.spacepoint_indices().push_back(sp);
      }
      seed.quality() = qualities[i];
    }
  }

  // Copy it to a host collection and check it.
  traccc::edm::long_seed_collection::host seeds{host_mr};
  copy(buffer, seeds)->wait();

  ASSERT_EQ(seeds.size(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto seed = seeds.at(i);
    ASSERT_EQ(seed.spacepoint_indices().size(), input[i].size());
    for (std::size_t j = 0; j < input[i].size(); ++j) {
      EXPECT_EQ(seed.spacepoint_indices()[j], input[i][j]);
    }
    EXPECT_FLOAT_EQ(seed.quality(), qualities[i]);
  }
}
