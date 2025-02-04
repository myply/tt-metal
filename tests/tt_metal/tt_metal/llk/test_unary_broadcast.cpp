// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include <tt-metalium/bfloat8.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::broadcast {

enum BroadcastDim : uint8_t { ROW = 0, COL = 1, SCALAR = 2, NONE = 3 };

const map<BroadcastDim, string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
    {BroadcastDim::NONE, "BroadcastType::NONE"}};

struct UnaryBroadcastConfig {
    BroadcastDim broadcast_dim;
    tt::DataFormat in0_t;
    tt::DataFormat out0_t;
};

// Assume 1Xn tiles.
template <class T>
std::vector<T> get_broadcasted_vec(std::vector<T>& src, const std::vector<uint32_t>& shape, BroadcastDim dim) {
    int num_tiles = shape.at(0);
    int num_rows = shape.at(1);
    int num_cols = shape.at(2);
    int tile_elem_count = num_rows * num_cols;

    std::vector<T> vBroadcast(num_tiles * num_cols * num_rows);

    if (dim == BroadcastDim::NONE) {
        vBroadcast = src;
    } else {
        for (int t = 0; t < num_tiles; t++) {
            int tile_offset = tile_elem_count * t;
            for (int i = 0; i < num_rows; i++) {
                for (int j = 0; j < num_cols; j++) {
                    T broadcast_value;
                    switch (dim) {
                        case BroadcastDim::ROW: {
                            broadcast_value = src[tile_offset + j];
                            break;
                        }
                        case BroadcastDim::COL: {
                            broadcast_value = src[tile_offset + (i * num_cols)];
                            break;
                        }
                        case BroadcastDim::SCALAR: {
                            broadcast_value = src[tile_offset];
                            break;
                        }
                        default: {
                            TT_THROW("Unsupported BroadcastDim={}", dim);
                            break;
                        }
                    }

                    vBroadcast[tile_offset + (i * num_cols + j)] = broadcast_value;
                }
            }
        }
    }

    return vBroadcast;
}

// T_in : type of src vector
// T_out : type of data the packer will pack out
// Assume nx1 tiles, row major data layout.
template <class T_in>
std::vector<uint32_t> get_tilized_packed_golden_broadcast(
    std::vector<T_in>& src, const std::vector<uint32_t>& shape, BroadcastDim dim, tt::DataFormat T_out) {
    static_assert(
        std::is_same<bfloat16, T_in>::value || std::is_same<float, T_in>::value,
        "Only float & Float_16b type as input allowed");
    std::vector<uint32_t> tilized_packed_res;
    unit_tests::compute::GoldenConfig config = {.num_tiles_r_dim = shape.at(0), .num_tiles_c_dim = 1};
    std::vector<T_in> vBroadcast = get_broadcasted_vec(src, shape, dim);
    if constexpr (std::is_same<bfloat16, T_in>::value) {
        if (T_out == tt::DataFormat::Float16_b) {
            auto packed_vec = pack_vector<uint32_t, bfloat16>(vBroadcast);
            tilized_packed_res = unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            std::vector<float> tempfp32v;
            tempfp32v.reserve(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp32v[i] = vBroadcast[i].to_float();
            }
            tilized_packed_res = pack_fp32_vec_as_bfp8_tiles(tempfp32v, true, false);
        }
    } else if constexpr (std::is_same<float, T_in>::value) {
        if (T_out == tt::DataFormat::Float16_b) {
            std::vector<bfloat16> tempfp16bv;
            tempfp16bv.reserve(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp16bv[i] = vBroadcast[i];
            }
            auto packed_vec = pack_vector<uint32_t, bfloat16>(tempfp16bv);
            tilized_packed_res = unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            tilized_packed_res = pack_fp32_vec_as_bfp8_tiles(vBroadcast, true, false);
        }
    }
    return tilized_packed_res;
}

auto GetBufferHelper(tt_metal::IDevice* device, tt::DataFormat dformat, uint32_t num_tiles) {
    uint32_t single_tile_size = tile_size(dformat);
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    return CreateBuffer(dram_config);
}

CBHandle CreateCircularBufferHelper(
    Program& program, CoreCoord& core, uint32_t num_pages, tt::DataFormat dformat, uint32_t id) {
    uint32_t page_size = tile_size(dformat);
    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(num_pages * page_size, {{id, dformat}}).set_page_size(id, page_size);
    return tt_metal::CreateCircularBuffer(program, core, l1_cb_config);
}

void get_packed_tilized_input_output_pair(
    tt::DataFormat in_t,
    tt::DataFormat out_t,
    uint32_t num_tiles,
    BroadcastDim bcast_dim,
    std::vector<uint32_t>& packed_tilized_input,
    std::vector<uint32_t>& packed_tilized_output) {
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t num_single_tile_elem = tile_width * tile_height;

    if (in_t == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> input = generate_uniform_random_vector<bfloat16>(
            -1.0f, 1.0f, num_tiles * num_single_tile_elem, std::chrono::system_clock::now().time_since_epoch().count());

        unit_tests::compute::GoldenConfig config = {.num_tiles_r_dim = num_tiles, .num_tiles_c_dim = 1};
        auto packed_input = pack_vector<uint32_t, bfloat16>(input);
        packed_tilized_input = unit_tests::compute::gold_standard_tilize(packed_input, config);
        packed_tilized_output =
            get_tilized_packed_golden_broadcast(input, {num_tiles, tile_width, tile_height}, bcast_dim, out_t);
    } else {
    }
}

void run_single_core_unary_broadcast(tt_metal::IDevice* device, const UnaryBroadcastConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t num_tiles = 32;
    constexpr uint32_t num_blocks = 4;
    constexpr uint32_t block_size = num_tiles / num_blocks;
    tt::DataFormat in0_t = test_config.in0_t;
    tt::DataFormat out0_t = test_config.out0_t;

    auto src_dram_buffer_0 = GetBufferHelper(device, in0_t, num_tiles);
    auto dst_dram_buffer_0 = GetBufferHelper(device, out0_t, num_tiles);
    auto l1_src_cb_0 = CreateCircularBufferHelper(program, core, block_size * 2, in0_t, 0);
    auto l1_dst_cb_0 = CreateCircularBufferHelper(program, core, block_size * 2, out0_t, 16);

    std::map<string, string> defines = {{"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)}};

    log_info("Testing UNARY BCAST_DIM={}", defines["BCAST_DIM"]);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {num_blocks, block_size}, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)(src_dram_buffer_0->address()),
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)(dst_dram_buffer_0->address()),
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    std::vector<uint32_t> packed_tilized_input_0, golden_packed_tilized_output_0;
    get_packed_tilized_input_output_pair(
        in0_t, out0_t, num_tiles, test_config.broadcast_dim, packed_tilized_input_0, golden_packed_tilized_output_0);
    tt_metal::detail::WriteToBuffer(src_dram_buffer_0, packed_tilized_input_0);
    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data_0;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_0, dest_buffer_data_0);

    bool result = is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data_0, golden_packed_tilized_output_0, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0);
        });

    ASSERT_TRUE(result);
}
}  // namespace unit_tests::compute::broadcast

class UnaryBroadcastParameterizedDeviceFixture
    : public DeviceFixture,
      public testing::WithParamInterface<unit_tests::compute::broadcast::UnaryBroadcastConfig> {};

TEST_P(UnaryBroadcastParameterizedDeviceFixture, TensixComputeSingleTileUnaryBroadcast) {
    if (this->arch_ == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    unit_tests::compute::broadcast::UnaryBroadcastConfig test_config = GetParam();
    unit_tests::compute::broadcast::run_single_core_unary_broadcast(this->devices_.at(0), test_config);
}

using namespace unit_tests::compute::broadcast;

INSTANTIATE_TEST_SUITE_P(
    ComputeSingleTileUnaryBroadcast,
    UnaryBroadcastParameterizedDeviceFixture,
    ::testing::Values(
        (UnaryBroadcastConfig){BroadcastDim::NONE, tt::DataFormat::Float16_b, tt::DataFormat::Float16_b},
        (UnaryBroadcastConfig){BroadcastDim::ROW, tt::DataFormat::Float16_b, tt::DataFormat::Float16_b},
        (UnaryBroadcastConfig){BroadcastDim::COL, tt::DataFormat::Float16_b, tt::DataFormat::Float16_b},
        (UnaryBroadcastConfig){BroadcastDim::SCALAR, tt::DataFormat::Float16_b, tt::DataFormat::Float16_b}));
