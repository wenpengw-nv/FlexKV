/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include "gds_layout_transform.cuh"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4*>(ptr)

/**
 * CUDA kernel for layout transformation between staging buffer and target GPU tensor
 * 
 * Each CUDA block handles one chunk (one layer's K or V for one block)
 * This achieves high throughput by utilizing GPU's internal memory bandwidth.
 */
template<BackendType Type>
__global__ void layout_transform_kernel(
    int64_t* staging_base,
    int64_t staging_layer_stride,
    int64_t staging_kv_stride,
    int64_t staging_block_stride,
    int64_t chunk_size,  // in int64_t units
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool staging_to_target
) {
    int kv_dim = is_mla ? 1 : 2;
    int num_chunks = num_layers * kv_dim * num_blocks;
    int64_t chunk_size_in_float4 = chunk_size * sizeof(int64_t) / sizeof(float4);
    
    for (int chunk_idx = blockIdx.x; chunk_idx < num_chunks; chunk_idx += gridDim.x) {
        // Decode chunk index
        int block_local_idx = chunk_idx % num_blocks;
        int layer_idx = chunk_idx / (num_blocks * kv_dim);
        int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
        
        int64_t gpu_block_idx = gpu_block_ids[block_local_idx];
        
        // Calculate staging buffer pointer
        // Staging layout: [block_local_idx][layer_idx][kv_idx][data...]
        int64_t* staging_ptr = staging_base + 
                               block_local_idx * staging_block_stride +
                               layer_idx * staging_layer_stride +
                               kv_idx * staging_kv_stride;
        
        // Calculate target GPU pointer using template specialization
        int64_t* gpu_ptr = ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
        
        // Determine source and destination
        int64_t* src_ptr = staging_to_target ? staging_ptr : gpu_ptr;
        int64_t* dst_ptr = staging_to_target ? gpu_ptr : staging_ptr;
        
        // Copy data using float4 for coalesced memory access
        for (int64_t idx = threadIdx.x; idx < chunk_size_in_float4; idx += blockDim.x) {
            float4 element = FLOAT4_PTR(src_ptr)[idx];
            FLOAT4_PTR(dst_ptr)[idx] = element;
        }
    }
}

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
    cudaStream_t stream
) {
    if (num_blocks == 0 || num_layers == 0) return;
    
    int block_size = 256;
    int kv_dim = is_mla ? 1 : 2;
    int num_chunks = num_layers * kv_dim * num_blocks;
    
    // Calculate optimal grid size
    int device_id;
    cudaGetDevice(&device_id);
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, layout_transform_kernel<Type>, block_size, 0);
    
    int grid_size = std::min(num_chunks, num_sms * max_blocks_per_sm);
    
    layout_transform_kernel<Type><<<grid_size, block_size, 0, stream>>>(
        staging_base,
        staging_layer_stride,
        staging_kv_stride,
        staging_block_stride,
        chunk_size,
        gpu_handler,
        gpu_block_ids,
        num_blocks,
        num_layers,
        is_mla,
        staging_to_target
    );
}

// Explicit template instantiations
template void launch_layout_transform_kernel<BackendType::VLLM>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

template void launch_layout_transform_kernel<BackendType::TRTLLM>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

template void launch_layout_transform_kernel<BackendType::SGLANG>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

} // namespace flexkv

