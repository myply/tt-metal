// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
void kernel_main() {

    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t num_tiles = 256;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat dst_data_format = get_dataformat(cb_id_in1);
    DPRINT << "write ok0: " << ENDL();
    cb_wait_front(cb_id_in1, num_tiles);
    DPRINT << "write ok1: " << ENDL();
    
    constexpr bool dst_is_dram = true;
    const InterleavedAddrGenFast<dst_is_dram> s0 = {
    .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};
    uint32_t l1_addr_in1 = get_write_ptr(cb_id_in1);
    uint32_t tile_id = 0;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_write_tile(tile_id, s0, l1_addr_in1);
        // when and why do we need barrier
        noc_async_write_barrier();
        tile_id += 1;
        l1_addr_in1 += dst_tile_bytes;
    }
    DPRINT << "write ok2: " << ENDL();
    cb_pop_front(cb_id_in1, num_tiles);
}