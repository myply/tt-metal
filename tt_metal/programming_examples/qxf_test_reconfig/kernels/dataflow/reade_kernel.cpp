// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
void kernel_main() {
    asm("nop");
    constexpr uint32_t cb_in0 = 0;
    cb_push_back(cb_in0, 256);
    asm("nop");
}