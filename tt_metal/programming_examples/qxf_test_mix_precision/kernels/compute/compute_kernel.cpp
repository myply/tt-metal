// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"
void transpose_block(uint32_t in_cb, uint32_t out_cb, uint32_t row_tiles, uint32_t col_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    uint32_t num_tiles=row_tiles*col_tiles;
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    uint32_t tile_idx = 0;
#pragma GCC unroll 0
    for (uint32_t i = 0; i < col_tiles; i++) {
        for(uint32_t j=0;j<row_tiles;j++){
            acquire_dst();
            copy_tile(in_cb, tile_idx, 0 /*dst*/);
            pack_tile(0, out_cb);
            release_dst();
            tile_idx += col_tiles;
        }
        tile_idx = tile_idx - num_tiles + 1;
    }
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in_cb, num_tiles);
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

//#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        MATH(DPRINT << "math ok1: " << ENDL();)
        acquire_dst();
        MATH(DPRINT << "math ok2: " << ENDL();)
        copy_tile(in_cb, i, 0 /*dst*/);
        MATH(DPRINT << "math ok3: " << ENDL();)
        pack_tile(0, out_cb);
        MATH(DPRINT << "math ok4: " << ENDL();)
        cb_push_back(out_cb, 1);
        release_dst();
        MATH(DPRINT << "math ok5: " << ENDL();)
    }
    cb_pop_front(in_cb, num_tiles);
}
namespace NAMESPACE {
void MAIN {
    // asm("nop");
    uint32_t num_tiles = 256;
    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    //when and why do we need this????
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_1);
    //mm_init();
    MATH(DPRINT << "math ok0: " << ENDL();)
    reconfig_data_format_srca(cb_in0);
    pack_reconfig_data_format(cb_in1);
    copy_block(cb_in0, cb_in1, num_tiles);
    MATH(DPRINT << "math ok6: " << ENDL();)

    // cb_pop_front(tt::CBIndex::c_0, num_tiles);
    // cb_push_back(tt::CBIndex::c_1, num_tiles);
}
}  // namespace NAMESPACE