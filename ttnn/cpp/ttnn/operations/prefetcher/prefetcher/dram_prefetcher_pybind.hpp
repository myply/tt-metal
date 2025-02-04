// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::dram_prefetcher::detail {

void bind_dram_prefetcher(pybind11::module& module);

}  // namespace ttnn::operations::dram_prefetcher::detail
