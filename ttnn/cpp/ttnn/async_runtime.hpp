// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/types.hpp"
#include "ttnn/run_operation.hpp"
#include "types.hpp"

namespace ttnn {
using queue_id = uint8_t;

void write_buffer(
    queue_id cq_id,
    Tensor& dst,
    std::vector<std::shared_ptr<void>> src,
    const std::optional<BufferRegion>& region = std::nullopt);

void read_buffer(
    queue_id cq_id,
    Tensor& src,
    std::vector<std::shared_ptr<void>> dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    size_t src_offset = 0,
    bool blocking = true);

void queue_synchronize(CommandQueue& cq);

void event_synchronize(const std::shared_ptr<Event>& event);

bool event_query(const std::shared_ptr<Event>& event);

void wait_for_event(CommandQueue& cq, const std::shared_ptr<Event>& event);

void record_event(CommandQueue& cq, const std::shared_ptr<Event>& event);
}  // namespace ttnn
