# GDS Staged Transfer 设计说明文档

## 1. 背景与问题

### 1.1 现有架构

FlexKV支持三种GPU KV Cache后端布局：

| Backend | GPU Layout | 特点 |
|---------|-----------|------|
| **TRTLLM** | Block-first | `[block][layer][kv][data]` |
| **VLLM** | Layer-first | 每层一个tensor，`tensor[layer][block][data]` |
| **SGLANG** | Layer-first | 每个kv和层一个tensor |

SSD文件布局统一为 **Block-first**：`[block][layer][kv][data]`

### 1.2 问题描述

GDS (GPU Direct Storage) 传输在连续大块数据传输时性能最佳。对于TRTLLM后端，由于GPU和SSD布局一致（都是block-first），可以实现：

- **单次GDS调用传输整个block的所有layers数据**
- **多线程并行传输不同blocks**

但对于VLLM和SGLANG后端，由于GPU布局是layer-first，只能逐layer传输：

- 每个layer需要单独的GDS调用
- 传输粒度小，无法充分利用GDS带宽

### 1.3 性能差距

```
TRTLLM:  一个block所有layers → 1次GDS调用 → 高带宽利用率
VLLM:    一个block的N个layers → N次GDS调用 → 低带宽利用率
```

---

## 2. 解决方案

### 2.1 核心思路

引入 **Staging Buffer** 作为中间层：

```
┌─────────────────────────────────────────────────────────────────┐
│                        READ 操作 (SSD → GPU)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    GDS (block-first)    ┌──────────────┐         │
│   │   SSD   │ ───────────────────────►│   Staging    │         │
│   │  Files  │     高带宽传输           │   Buffer     │         │
│   └─────────┘                         └──────┬───────┘         │
│                                              │                  │
│                                   CUDA Kernel│                  │
│                                   Layout变换 │                  │
│                                              ▼                  │
│                                       ┌────────────┐           │
│                                       │  Target    │           │
│                                       │  GPU       │           │
│                                       │  Tensor    │           │
│                                       └────────────┘           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       WRITE 操作 (GPU → SSD)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────┐                                               │
│   │  Source    │                                               │
│   │  GPU       │                                               │
│   │  Tensor    │                                               │
│   └─────┬──────┘                                               │
│         │                                                       │
│         │ CUDA Kernel                                          │
│         │ Layout变换                                            │
│         ▼                                                       │
│   ┌──────────────┐    GDS (block-first)    ┌─────────┐         │
│   │   Staging    │ ───────────────────────►│   SSD   │         │
│   │   Buffer     │     高带宽传输           │  Files  │         │
│   └──────────────┘                         └─────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么这个方案有效

1. **GDS传输优化**：Staging buffer采用block-first布局，与SSD一致，单次传输整个block
2. **GPU内存带宽高**：现代GPU内存带宽>2TB/s，layout变换开销可忽略
3. **对TRTLLM无影响**：TRTLLM直接使用原有路径，无额外开销

---

## 3. 实现细节

### 3.1 文件结构

```
csrc/gds/
├── gds_manager.h              # 新增 transfer_kv_blocks_gds_staged 声明
├── gds_manager.cpp            # 新增 staged transfer 实现
├── gds_layout_transform.cuh   # [新文件] Layout变换kernel头文件
└── gds_layout_transform.cu    # [新文件] Layout变换kernel实现

flexkv/
├── common/config.py           # 新增环境变量配置
└── transfer/worker.py         # 更新GDSTransferWorker使用staged transfer
```

### 3.2 Layout变换Kernel

```cpp
// gds_layout_transform.cu
template<BackendType Type>
__global__ void layout_transform_kernel(
    int64_t* staging_base,           // Staging buffer (block-first)
    int64_t staging_layer_stride,
    int64_t staging_kv_stride,
    int64_t staging_block_stride,
    int64_t chunk_size,
    GTensorHandler gpu_handler,       // Target GPU tensor handler
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool staging_to_target           // 方向控制
);
```

**特点：**
- 使用 `float4` 向量化访问（16字节对齐）
- 模板特化支持不同backend的指针计算
- Grid/Block大小自动优化

### 3.3 Staged Transfer函数

```cpp
// gds_manager.cpp
template<BackendType Type>
void transfer_kv_blocks_gds_staged(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    void* staging_buffer,    // 可选：预分配的staging buffer
    bool verbose,
    bool is_mla
);
```

**执行流程：**

```cpp
// 对于TRTLLM：直接走原有路径
if constexpr (Type == BackendType::TRTLLM) {
    transfer_kv_blocks_gds<Type>(...);
    return;
}

// 对于VLLM/SGLANG：
if (!is_read) {
    // WRITE: GPU → Staging (CUDA kernel)
    launch_layout_transform_kernel<Type>(..., staging_to_target=false);
}

// GDS传输：Staging ↔ SSD (多线程并行)
for (device : devices) {
    for (block : blocks_on_device) {
        enqueue_task([&]() {
            if (is_read) {
                gds_manager.read(file, staging_ptr, size, offset);
            } else {
                gds_manager.write(file, staging_ptr, size, offset);
            }
        });
    }
}
wait_all_tasks();

if (is_read) {
    // READ: Staging → GPU (CUDA kernel)
    launch_layout_transform_kernel<Type>(..., staging_to_target=true);
}
```

### 3.4 Python集成

```python
# worker.py - GDSTransferWorker
class GDSTransferWorker(TransferWorkerBase):
    def __init__(self, ..., 
                 use_staged_transfer: Optional[bool] = None,
                 max_staging_blocks: Optional[int] = None):
        # 从环境变量读取默认值
        if use_staged_transfer is None:
            use_staged_transfer = GLOBAL_CONFIG_FROM_ENV.gds_use_staged_transfer
        if max_staging_blocks is None:
            max_staging_blocks = GLOBAL_CONFIG_FROM_ENV.gds_max_staging_blocks
        
        # TRTLLM不需要staged transfer
        self.use_staged_transfer = use_staged_transfer and (self.gpu_block_type_ != 1)
        
        # 预分配staging buffer
        if self.use_staged_transfer:
            self.staging_buffer = torch.empty(
                staging_total_size, dtype=torch.uint8, device=f"cuda:{gpu_device_id}"
            )
```

---

## 4. 配置选项

### 4.1 环境变量

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `FLEXKV_GDS_USE_STAGED_TRANSFER` | `1` | 是否启用staged transfer |
| `FLEXKV_GDS_MAX_STAGING_BLOCKS` | `64` | Staging buffer最大block数 |

### 4.2 使用示例

```bash
# 默认启用（推荐）
python your_app.py

# 禁用staged transfer
export FLEXKV_GDS_USE_STAGED_TRANSFER=0
python your_app.py

# 增大staging buffer
export FLEXKV_GDS_MAX_STAGING_BLOCKS=128
python your_app.py
```

---

## 5. 内存开销分析

### 5.1 Staging Buffer大小计算

```
staging_buffer_size = chunk_size × kv_dim × num_layers × max_staging_blocks

示例（Llama-70B, FP16）:
- chunk_size ≈ 128KB (per layer per KV)
- kv_dim = 2 (K + V)
- num_layers = 80
- max_staging_blocks = 64

staging_buffer_size = 128KB × 2 × 80 × 64 = 1.25GB
```

### 5.2 内存优化

1. **预分配复用**：Staging buffer在worker初始化时分配一次，多次传输复用
2. **动态回退**：当传输blocks数超过`max_staging_blocks`时，自动分配临时buffer
3. **按需分配**：TRTLLM后端不分配staging buffer

---

## 6. 性能预期

| 场景 | 原方案 | Staged Transfer |
|-----|-------|-----------------|
| TRTLLM | N/A（已是最优） | 无变化 |
| VLLM (80层) | 80次GDS调用/block | 1次GDS调用/block + kernel |
| SGLANG (80层) | 160次GDS调用/block | 1次GDS调用/block + kernel |

**预期收益：**
- GDS调用次数减少80-160倍
- 整体传输带宽提升（取决于具体硬件配置）
- Layout变换开销可忽略（GPU内存带宽>>SSD带宽）

---

## 7. 兼容性

- **向后兼容**：现有代码无需修改即可使用
- **Backend支持**：VLLM、SGLANG、TRTLLM
- **可选特性**：可通过环境变量禁用

---

## 8. API参考

### 8.1 C++ API

```cpp
// 新增函数 (gds_manager.h)
namespace flexkv {

template<BackendType Type>
void transfer_kv_blocks_gds_staged(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    GTensorHandler gpu_tensor_handler,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    void* staging_buffer = nullptr,  // nullptr则内部分配
    bool verbose = false,
    bool is_mla = false
);

} // namespace flexkv
```

### 8.2 Python API

```python
# 新增绑定 (c_ext module)
flexkv.c_ext.transfer_kv_blocks_gds_staged(
    gds_manager,           # GDSManager实例
    gpu_layer_id_list,     # int32 tensor
    gpu_layer_ptrs_tensor, # int64 tensor
    gds_block_ids,         # int64 tensor
    gpu_block_ids,         # int64 tensor
    gpu_kv_stride_in_bytes,
    gpu_block_stride_in_bytes,
    gpu_layer_stride_in_bytes,
    gds_layer_stride_in_bytes,
    gds_block_stride_in_bytes,
    gds_kv_stride_in_bytes,
    block_size_in_bytes,
    gds_copy_off_inside_chunks,
    num_blocks_per_file,
    total_layers,
    is_read,
    verbose=False,
    is_mla=False,
    gpu_block_type=0,      # 0=VLLM, 1=TRTLLM, 2=SGLANG
    staging_buffer=None    # 可选：预分配的staging buffer tensor
)
```

