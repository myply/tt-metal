// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
// void kernel_main() {
//     asm("nop");
//     constexpr uint32_t cb_in0 = 0;
//     cb_push_back(cb_in0, 256);
//     asm("nop");
// }
void kernel_main() {

    uint32_t src0_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t num_tiles = 256;
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    cb_reserve_back(cb_id_in0, num_tiles);

    
    constexpr bool src0_is_dram = true;
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
    .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
    uint32_t l1_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t tile_id = 0;
    DPRINT << "read ok0: " << ENDL();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read_tile(tile_id, s0, l1_addr_in0);
        // when and why do we need barrier
        noc_async_read_barrier();
        tile_id += 1;
        l1_addr_in0 += src0_tile_bytes;
    }
    DPRINT << "read ok1: " << ENDL();
    cb_push_back(cb_id_in0, num_tiles);
}