/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>
#include "../gtensor_handler.cuh"

namespace flexkv {

/**
 * Transform layout between staging buffer (block-first) and target GPU tensor
 * 
 * Staging buffer layout: [block_idx][layer_idx][kv_idx][data...]
 * The staging buffer has the same layout as SSD files (block-first).
 * 
 * @param staging_base Base pointer of the staging buffer
 * @param staging_layer_stride Stride between layers in staging buffer (in int64_t units)
 * @param staging_kv_stride Stride between K and V in staging buffer (in int64_t units)
 * @param staging_block_stride Stride between blocks in staging buffer (in int64_t units)
 * @param chunk_size Size of each chunk (in int64_t units)
 * @param gpu_handler Target GPU tensor handler
 * @param gpu_block_ids Array of target GPU block IDs
 * @param num_blocks Number of blocks to transform
 * @param num_layers Number of layers
 * @param is_mla Whether using MLA (single KV instead of K+V)
 * @param staging_to_target Direction: true = staging->target, false = target->staging
 */
template<BackendType Type>
void launch_layout_transform_kernel(
    int64_t* staging_base,
    int64_t staging_layer_stride,
    int64_t staging_kv_stride,
    int64_t staging_block_stride,
    int64_t chunk_size,
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool staging_to_target,
    cudaStream_t stream);

} // namespace flexkv

