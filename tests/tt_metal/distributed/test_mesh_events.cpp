// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "tests/tt_metal/tt_metal/dispatch/dispatch_test_utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/distributed/utils.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshEventsTestT3000 = T3000MultiCQMeshDeviceFixture;
using MeshEventsTestSuite = GenericMultiCQMeshDeviceFixture;

TEST_F(MeshEventsTestSuite, ReplicatedAsyncIO) {
    uint32_t NUM_TILES = 1000;
    uint32_t num_iterations = 20;
    int32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    ReplicatedBufferConfig global_buffer_config = {
        .size = NUM_TILES * single_tile_size,
    };

    std::shared_ptr<MeshBuffer> buf =
        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec(NUM_TILES * single_tile_size / sizeof(uint32_t), 0);
        std::iota(src_vec.begin(), src_vec.end(), i);

        std::vector<std::vector<uint32_t>> readback_vecs = {};
        std::shared_ptr<MeshEvent> event = std::make_shared<MeshEvent>();
        // Writes on CQ 0
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(0), buf, src_vec);
        // Device to Device Synchronization
        EnqueueRecordEvent(mesh_device_->mesh_command_queue(0), event);
        EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), event);

        // Reads on CQ 1
        for (std::size_t logical_x = 0; logical_x < buf->device()->num_cols(); logical_x++) {
            for (std::size_t logical_y = 0; logical_y < buf->device()->num_rows(); logical_y++) {
                readback_vecs.push_back({});
                auto shard = buf->get_device_buffer(Coordinate(logical_y, logical_x));
                ReadShard(
                    mesh_device_->mesh_command_queue(1), readback_vecs.back(), buf, Coordinate(logical_y, logical_x));
            }
        }

        for (auto& vec : readback_vecs) {
            EXPECT_EQ(vec, src_vec);
        }
    }
}

TEST_F(MeshEventsTestT3000, ShardedAsyncIO) {
    uint32_t num_iterations = 20;
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    Shape2D global_buffer_shape = {2048, 2048};
    Shape2D shard_shape = {512, 1024};

    uint32_t global_buffer_size = global_buffer_shape.height() * global_buffer_shape.width() * sizeof(uint32_t);

    ShardedBufferConfig sharded_config{
        .global_size = global_buffer_size,
        .global_buffer_shape = global_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer = MeshBuffer::create(sharded_config, per_device_buffer_config, mesh_device_.get());
    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec =
            std::vector<uint32_t>(global_buffer_shape.height() * global_buffer_shape.width(), 0);
        std::iota(src_vec.begin(), src_vec.end(), i);
        std::shared_ptr<MeshEvent> event = std::make_shared<MeshEvent>();
        // Writes on CQ 0
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(0), mesh_buffer, src_vec);
        if (i % 2) {
            // Test Host <-> Device synchronization
            EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(0), event);
            EventSynchronize(event);
        } else {
            // Test Device <-> Device synchronization
            EnqueueRecordEvent(mesh_device_->mesh_command_queue(0), event);
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), event);
        }
        // Reads on CQ 1
        std::vector<uint32_t> dst_vec = {};
        EnqueueReadMeshBuffer(mesh_device_->mesh_command_queue(1), dst_vec, mesh_buffer);

        EXPECT_EQ(dst_vec, src_vec);
    }
}

TEST_F(MeshEventsTestSuite, AsyncWorkloadAndIO) {
    uint32_t num_iters = 5;
    std::vector<std::shared_ptr<MeshBuffer>> src0_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> src1_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    auto programs = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, src0_bufs, src1_bufs, output_bufs);
    auto mesh_workload = CreateMeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, 0});
    LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {mesh_device_->num_cols() - 1, 1});

    AddProgramToMeshWorkload(mesh_workload, *programs[0], devices_0);
    AddProgramToMeshWorkload(mesh_workload, *programs[1], devices_1);

    for (int iter = 0; iter < num_iters; iter++) {
        std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), iter + 2);
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src1_bufs[0]->size(), iter + 3);

        std::shared_ptr<MeshEvent> write_event = std::make_shared<MeshEvent>();
        std::shared_ptr<MeshEvent> op_event = std::make_shared<MeshEvent>();

        // Issue writes on MeshCQ 1
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteMeshBuffer(
                    mesh_device_->mesh_command_queue(1), src0_bufs[col_idx * worker_grid_size.y + row_idx], src0_vec);
                EnqueueWriteMeshBuffer(
                    mesh_device_->mesh_command_queue(1), src1_bufs[col_idx * worker_grid_size.y + row_idx], src1_vec);
            }
        }
        if (iter % 2) {
            // Test Host <-> Device Synchronization
            EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(1), write_event);
            EventSynchronize(write_event);
        } else {
            // Test Device <-> Device Synchronization
            EnqueueRecordEvent(mesh_device_->mesh_command_queue(1), write_event);
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(0), write_event);
        }
        // Issue workloads on MeshCQ 0
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(0), mesh_workload, false);
        if (iter % 2) {
            // Test Device <-> Device Synchronization
            EnqueueRecordEvent(mesh_device_->mesh_command_queue(0), op_event);
            EnqueueWaitForEvent(mesh_device_->mesh_command_queue(1), op_event);
        } else {
            // Test Host <-> Device Synchronization
            EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(0), op_event);
            EventSynchronize(op_event);
        }

        // Issue reads on MeshCQ 1
        for (std::size_t logical_y = 0; logical_y < mesh_device_->num_rows(); logical_y++) {
            for (std::size_t logical_x = 0; logical_x < mesh_device_->num_cols(); logical_x++) {
                for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                    for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                        std::vector<bfloat16> dst_vec = {};
                        ReadShard(
                            mesh_device_->mesh_command_queue(1),
                            dst_vec,
                            output_bufs[col_idx * worker_grid_size.y + row_idx],
                            Coordinate(logical_y, logical_x));
                        if (logical_y == 0) {
                            for (int i = 0; i < dst_vec.size(); i++) {
                                EXPECT_EQ(dst_vec[i].to_float(), (2 * iter + 5));
                            }
                        } else {
                            for (int i = 0; i < dst_vec.size(); i++) {
                                EXPECT_EQ(dst_vec[i].to_float(), (iter + 2) * (iter + 3));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MeshEventsTestSuite, CustomDeviceRanges) {
    uint32_t NUM_TILES = 1000;
    uint32_t num_iterations = 20;
    int32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = false};
    ReplicatedBufferConfig global_buffer_config = {
        .size = NUM_TILES * single_tile_size,
    };

    std::shared_ptr<MeshBuffer> buf =
        MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    for (std::size_t i = 0; i < num_iterations; i++) {
        std::vector<uint32_t> src_vec(NUM_TILES * single_tile_size / sizeof(uint32_t), i);
        std::iota(src_vec.begin(), src_vec.end(), i);
        LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, 0});
        LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {mesh_device_->num_cols() - 1, 1});

        std::vector<std::vector<uint32_t>> readback_vecs = {};
        std::shared_ptr<MeshEvent> event_0 = std::make_shared<MeshEvent>();
        std::shared_ptr<MeshEvent> event_1 = std::make_shared<MeshEvent>();

        mesh_device_->mesh_command_queue(1).enqueue_write_shard_to_sub_grid(*buf, src_vec.data(), devices_0, false);
        EnqueueRecordEvent(mesh_device_->mesh_command_queue(1), event_0, {}, devices_0);
        EnqueueWaitForEvent(mesh_device_->mesh_command_queue(0), event_0);

        for (std::size_t logical_x = devices_0.start_coord.x; logical_x < devices_0.end_coord.x; logical_x++) {
            for (std::size_t logical_y = devices_0.start_coord.y; logical_y < devices_0.end_coord.y; logical_y++) {
                readback_vecs.push_back({});
                auto shard = buf->get_device_buffer(Coordinate(logical_y, logical_x));
                ReadShard(
                    mesh_device_->mesh_command_queue(0), readback_vecs.back(), buf, Coordinate(logical_y, logical_x));
            }
        }

        mesh_device_->mesh_command_queue(1).enqueue_write_shard_to_sub_grid(*buf, src_vec.data(), devices_1, false);
        EnqueueRecordEventToHost(mesh_device_->mesh_command_queue(1), event_1, {}, devices_1);
        EventSynchronize(event_1);

        for (std::size_t logical_x = devices_1.start_coord.x; logical_x < devices_1.end_coord.x; logical_x++) {
            for (std::size_t logical_y = devices_1.start_coord.y; logical_y < devices_1.end_coord.y; logical_y++) {
                readback_vecs.push_back({});
                auto shard = buf->get_device_buffer(Coordinate(logical_y, logical_x));
                ReadShard(
                    mesh_device_->mesh_command_queue(0), readback_vecs.back(), buf, Coordinate(logical_y, logical_x));
            }
        }
        for (auto& vec : readback_vecs) {
            EXPECT_EQ(vec, src_vec);
        }
    }
    Finish(mesh_device_->mesh_command_queue(0));
    Finish(mesh_device_->mesh_command_queue(1));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
