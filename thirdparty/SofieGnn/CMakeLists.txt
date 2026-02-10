# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.25)

# CMake include(s).
include(FetchContent)

# Tell the user what's happening.
message(STATUS "Building SofieGnn as part of the Acts project")

# Declare where to get SofieGnn from.
FetchContent_Declare(SofieGnn SYSTEM ${ACTS_SOFIEGNN_SOURCE})

set(SOFIE_GNN_USE_LARGE OFF)
set(SOFIE_GNN_DEBUG OFF)

# Now set up its build.
FetchContent_MakeAvailable(SofieGnn)
