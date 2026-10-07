// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Project include(s).
#include "traccc/edm/seed_collection.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Google test include(s).
#include <gtest/gtest.h>

// System include(s).
#include <array>
#include <compare>

TEST(SeedCollection, TripletAndLongSeeds) {
  vecmem::host_memory_resource host_mr;
  vecmem::copy copy;

  traccc::edm::seed_collection::buffer buffer(
      4u, host_mr, vecmem::data::buffer_type::resizable);
  copy.setup(buffer)->wait();

  // Fill it the way a kernel would: a triplet, a seed of five spacepoints and
  // one of the largest size.
  std::array<unsigned int, traccc::edm::seed_max_spacepoints> longest{};
  for (unsigned int i = 0; i < longest.size(); ++i) {
    longest[i] = 100u + i;
  }
  {
    traccc::edm::seed_collection::device seeds(buffer);
    seeds.push_back({{4, 8, 15}, 3u, 1.5f});
    seeds.push_back({{23, 42, 7, 9, 11}, 5u, 2.5f});
    seeds.push_back({longest, traccc::edm::seed_max_spacepoints, 3.5f});
  }

  traccc::edm::seed_collection::host seeds{host_mr};
  copy(buffer, seeds)->wait();
  ASSERT_EQ(seeds.size(), 3u);

  // A triplet reads as before.
  EXPECT_EQ(seeds.at(0).n_spacepoints(), 3u);
  EXPECT_EQ(seeds.at(0).bottom_index(), 4u);
  EXPECT_EQ(seeds.at(0).middle_index(), 8u);
  EXPECT_EQ(seeds.at(0).top_index(), 15u);
  EXPECT_FLOAT_EQ(seeds.at(0).quality(), 1.5f);

  // A long seed keeps all its spacepoints, innermost first, and reads as the
  // innermost, middle and outermost one.
  EXPECT_EQ(seeds.at(1).n_spacepoints(), 5u);
  const std::array<unsigned int, 5> expected = {23, 42, 7, 9, 11};
  for (unsigned int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(seeds.at(1).spacepoint_indices()[i], expected[i]);
  }
  EXPECT_EQ(seeds.at(1).bottom_index(), 23u);
  EXPECT_EQ(seeds.at(1).middle_index(), 7u);
  EXPECT_EQ(seeds.at(1).top_index(), 11u);

  EXPECT_EQ(seeds.at(2).n_spacepoints(), traccc::edm::seed_max_spacepoints);
  EXPECT_EQ(seeds.at(2).top_index(),
            100u + traccc::edm::seed_max_spacepoints - 1u);

  // Seeds compare on all their spacepoints.
  EXPECT_TRUE(seeds.at(0) == seeds.at(0));
  EXPECT_FALSE(seeds.at(0) == seeds.at(1));
  EXPECT_TRUE(std::is_eq(seeds.at(0) <=> seeds.at(0)));
}
