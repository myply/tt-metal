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

    uint32_t N=256;

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t single_tile_size = 2 * 1024;


    //DRAM Data
    uint32_t dram_buffer_A_size =single_tile_size * N;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    uint32_t dram_buffer_DST_size =single_tile_size * N; 
    tt_metal::InterleavedBufferConfig dram_config_DST{
        .device = device,
        .size = dram_buffer_DST_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_DST);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    
    
    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = N;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    uint32_t dst_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{dst_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dst_cb_index, single_tile_size);
    auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst_config);



    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_mix_precision/kernels/dataflow/reade_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_mix_precision/kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};
    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/qxf_test_mix_precision/kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });



    // /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(program,binary_reader_kernel_id,core,{src0_addr});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_addr});


    std::vector<bfloat16> a(N*1024,131.4f);
    std::vector<bfloat16> out(N*1024,0.0f);
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, out.data(), true);
    Finish(cq);
    for (uint32_t i=1024;i<2048;i++){
        std::cout<<out[i].to_float()<<std::endl;
    }
    CloseDevice(device);
    return 0;
}
