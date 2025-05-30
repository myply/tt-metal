// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// 
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    IDevice* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t single_tile_size = 2 * 1024;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 256;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    uint32_t output_cb_index = tt::CBIndex::c_1;
    uint32_t num_output_tiles = 256;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);


    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_reconfig/kernels/dataflow/reade_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_reconfig/kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};
    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_reconfig/kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });



    // /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(program,binary_reader_kernel_id,core,{});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    CloseDevice(device);
    return 0;
}
