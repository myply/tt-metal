// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <strong_type.hpp>

namespace ttnn {
/*
    Type must be moved to metal.
    Background:
    We have two software command queues available to overlap some work and reduce latency.
    For example, Op2 can be prepared in a different queue while the first queue is blocked, waiting for data readout by
    Op1. TT-NN operations allow specifying which queue should be used. The default queue is 0, and the possible values
    are 0 and 1.
*/
using QueueId = tt::stl::StrongType<uint8_t, struct QueueIdTag>;
static const QueueId DefaultQueueId = QueueId(0);

}  // namespace ttnn

// Exporting to tt::tt_metal namespace because ttnn
// defines some of its own types (think Tensor) in tt::tt_metal namespace.
namespace tt::tt_metal {
using QueueId = ttnn::QueueId;
}
