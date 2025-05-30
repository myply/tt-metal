// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm.hpp"

#include "ttnn/operations/moreh/moreh_layer_norm/device/moreh_layer_norm_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_layer_norm {
std::vector<std::optional<Tensor>> MorehLayerNorm::invoke(
    const Tensor& input,
    const uint32_t normalized_dims,
    const float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_layer_norm(
        input, normalized_dims, eps, gamma, beta, output, mean, rstd, memory_config, compute_kernel_config);
}

OptionalTensors MorehLayerNorm::create_async_optional_output_tensors(
    const Tensor& input,
    const uint32_t normalized_dims,
    const float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto return_mean = mean.has_value();
    const auto return_rstd = rstd.has_value();

    return {
        std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta})),
        return_mean ? std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta}))
                    : std::nullopt,
        return_rstd ? std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta}))
                    : std::nullopt};
}
}  // namespace ttnn::operations::moreh::moreh_layer_norm
