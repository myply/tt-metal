# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tests.tt_metal.microbenchmarks.ethernet.test_ethernet_link_write_worker_with_transaction_id_common import (
    profile_results,
)

from models.utility_functions import is_grayskull

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_write_worker_bandwidth.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


def run_erisc_write_worker_bw(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid, file_name
):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    test_latency = 0
    sample_size = sample_size_expected_bw[0]
    sample_size_expected_bw = sample_size_expected_bw[1]
    expected_bw_lower_bound = sample_size_expected_bw - 0.5
    expected_bw_upper_bound = sample_size_expected_bw + 0.5

    ARCH_NAME = os.getenv("ARCH_NAME")
    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_write_worker_latency_no_edm_{ARCH_NAME} \
                {sample_count} \
                {sample_size} \
                {channel_count} \
                {num_directions} \
                {test_latency} \
                {enable_worker} \
                {disable_trid}"
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    main_loop_latency = profile_results(
        sample_size, sample_count, channel_count, num_directions, test_latency, file_name
    )
    main_loop_bw = sample_size / main_loop_latency
    logger.info(f"sender_loop_latency {main_loop_latency}")
    logger.info(f"sender_loop_bw {main_loop_bw}")

    assert expected_bw_lower_bound <= main_loop_bw <= expected_bw_upper_bound


##################################### BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.21), (128, 1.72), (256, 3.44), (512, 6.89), (1024, 11.73), (2048, 11.83), (4096, 12.04), (8192, 12.07)],
)
def test_erisc_write_worker_bw_uni_dir(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.13), (128, 1.03), (256, 2.08), (512, 4.15), (1024, 8.31), (2048, 11.40), (4096, 11.82)],
)
def test_erisc_write_worker_bw_bi_dir(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )


##################################### No Worker BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("enable_worker", [0])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.28), (128, 2.25), (256, 4.39), (512, 8.35), (1024, 11.74), (2048, 11.84), (4096, 12.04), (8192, 12.07)],
)
def test_erisc_bw_uni_dir(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )


# bi-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("enable_worker", [0])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.19), (128, 1.59), (256, 3.19), (512, 6.39), (1024, 10.9), (2048, 11.4), (4096, 11.82)],
)
def test_erisc_bw_bi_dir(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )


##################################### No Transaction ID BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize("disable_trid", [1])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.18), (128, 1.46), (256, 2.93), (512, 5.73), (1024, 9.15), (2048, 11.83), (4096, 12.04), (8192, 12.07)],
)
def test_erisc_write_worker_bw_uni_dir_no_trid(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize("enable_worker", [1])
@pytest.mark.parametrize("disable_trid", [1])
@pytest.mark.parametrize(
    "sample_size_expected_bw",
    [(16, 0.10), (128, 0.87), (256, 1.73), (512, 3.44), (1024, 5.99), (2048, 9.70), (4096, 11.82)],
)
def test_erisc_write_worker_bw_bi_dir_no_trid(
    sample_count, sample_size_expected_bw, channel_count, num_directions, enable_worker, disable_trid
):
    run_erisc_write_worker_bw(
        sample_count,
        sample_size_expected_bw,
        channel_count,
        num_directions,
        enable_worker,
        disable_trid,
        FILE_NAME,
    )
