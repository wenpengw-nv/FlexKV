#!/usr/bin/env python3
"""
生成 Mooncake Trace 格式的 benchmark 数据集
用于 aiperf + TensorRT-LLM + FlexKV 压测

支持两种模式:

  Mode A: hash_ids (默认, 轻量级)
    - trace 只包含元数据 (hash_ids + input_length)
    - aiperf 内部从 Shakespeare 语料库生成 token 内容
    - 优点: 生成速度快, 文件小
    - 缺点: 实际前缀内容种类受限于语料库大小 (~20-30 万种)
    - 需要 aiperf --isl-block-size 150

  Mode B: text_input (精确控制, 需要 tokenizer)
    - trace 包含实际 prompt 文本 (text_input)
    - 使用指定 tokenizer 生成并 decode 真实 token 序列
    - 优点: 保证 2,000,000 种真正不同的前缀内容
    - 缺点: 生成较慢, 文件较大 (~5-7x), 需要安装 transformers
    - 不需要 aiperf --isl-block-size

Workload 描述:
  - 模型: gemma3-4b 纯文本
  - ISL: 160, OSL: 1
  - 前 150 token: 从 2,000,000 个不同 prompt 中随机选择
  - 后 10 token: 完全随机（每个请求都不同）
  - 单卡 QPS: 3000, 压测时间: 2 小时

用法:
  # Mode A: hash_ids (默认)
  python generate_trace_data.py --mode hash_ids --gpu_num 8

  # Mode B: text_input (精确 2M 种不同内容)
  python generate_trace_data.py --mode text_input --gpu_num 8 \\
    --tokenizer_path /raid/model/gemma-3-4b-it

  # 验证
  python generate_trace_data.py --verify
"""

import argparse
import json
import os
import sys
import time
import numpy as np


# ============================================================================
#  Mode A: hash_ids
# ============================================================================

def generate_hash_ids_mode(
    gpu_num: int,
    output_path: str,
    qps_per_gpu: int = 3000,
    duration_hours: float = 2.0,
    num_unique_prompts: int = 2_000_000,
    prefix_length: int = 150,
    suffix_length: int = 10,
    osl: int = 1,
    seed: int = 42,
):
    """
    Mode A: hash_ids 模式

    每条请求:
      {"timestamp": T, "input_length": 160, "output_length": 1,
       "hash_ids": [prefix_id, suffix_id]}

    aiperf 使用 block_size=prefix_length 时:
      hash_ids[0] → prefix_length tokens (共享, aiperf._cache 命中)
      hash_ids[1] → suffix_length tokens  (唯一, aiperf._cache miss)

    注意: 实际 token 内容多样性受限于 aiperf Shakespeare 语料库 (~20-30 万种)
    """
    np.random.seed(seed)

    isl = prefix_length + suffix_length
    total_qps = qps_per_gpu * gpu_num
    duration_seconds = duration_hours * 3600
    total_requests = int(total_qps * duration_seconds)
    mean_interval_ms = 1000.0 / total_qps
    suffix_id_start = num_unique_prompts

    _print_config("hash_ids", gpu_num, qps_per_gpu, total_qps, duration_hours,
                  duration_seconds, total_requests, num_unique_prompts, isl,
                  prefix_length, suffix_length, osl, mean_interval_ms, seed)
    print(f"  hash_ids 结构:        [prefix_id, suffix_id]")
    print(f"    prefix_id 范围:     [0, {num_unique_prompts:,})")
    print(f"    suffix_id 范围:     [{suffix_id_start:,}, {suffix_id_start + total_requests:,})")
    print(f"  aiperf --isl-block-size: {prefix_length} (必须设置! 默认512会报错)")
    print("=" * 70)

    estimated_size_gb = total_requests * 120 / (1024 ** 3)
    print(f"\n  预估文件大小: ~{estimated_size_gb:.2f} GB")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"trace_hashids_gpu{gpu_num}_qps{total_qps}_isl{isl}_osl{osl}_{duration_hours}h.jsonl",
    )
    print(f"  输出文件:     {output_file}\n")

    chunk_size = 1_000_000
    num_chunks = (total_requests + chunk_size - 1) // chunk_size
    current_ts = 0.0
    written = 0
    suffix_id = suffix_id_start
    t0 = time.time()

    with open(output_file, "w", encoding="utf-8") as f:
        for ci in range(num_chunks):
            bs = min(chunk_size, total_requests - ci * chunk_size)

            intervals = np.random.exponential(mean_interval_ms, bs)
            timestamps = np.cumsum(intervals) + current_ts
            current_ts = timestamps[-1]
            prefix_ids = np.random.randint(0, num_unique_prompts, bs)

            lines = []
            for i in range(bs):
                lines.append(json.dumps({
                    "timestamp": int(timestamps[i]),
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": [int(prefix_ids[i]), suffix_id],
                }, separators=(",", ":")))
                suffix_id += 1

            f.write("\n".join(lines))
            if ci < num_chunks - 1:
                f.write("\n")

            written += bs
            _print_progress(written, total_requests, t0)

    _print_done_hash_ids(output_file, written, current_ts, t0,
                         prefix_length, suffix_length, num_unique_prompts)
    return output_file


# ============================================================================
#  Mode B: text_input
# ============================================================================

def generate_text_input_mode(
    gpu_num: int,
    output_path: str,
    tokenizer_path: str,
    qps_per_gpu: int = 3000,
    duration_hours: float = 2.0,
    num_unique_prompts: int = 2_000_000,
    prefix_length: int = 150,
    suffix_length: int = 10,
    osl: int = 1,
    seed: int = 42,
):
    """
    Mode B: text_input 模式

    每条请求:
      {"timestamp": T, "text_input": "实际prompt文本...", "output_length": 1}

    原理:
      1. 用 tokenizer 预生成 2M 条唯一的 150-token 前缀 → decode 为文本 → prefix_texts[]
      2. 每条请求: 随机 10 个 suffix token → decode 为文本 → suffix_text
      3. text_input = prefix_texts[prefix_id] + suffix_text
      4. aiperf 直接使用 text_input 作为 prompt, 不走 hash_ids 逻辑

    保证: 2,000,000 种真正不同的前缀内容 (由 tokenizer vocab 的多样性保证)
    """
    from transformers import AutoTokenizer

    np.random.seed(seed)
    token_rng = np.random.RandomState(seed + 1)

    isl = prefix_length + suffix_length
    total_qps = qps_per_gpu * gpu_num
    duration_seconds = duration_hours * 3600
    total_requests = int(total_qps * duration_seconds)
    mean_interval_ms = 1000.0 / total_qps

    _print_config("text_input", gpu_num, qps_per_gpu, total_qps, duration_hours,
                  duration_seconds, total_requests, num_unique_prompts, isl,
                  prefix_length, suffix_length, osl, mean_interval_ms, seed)
    print(f"  tokenizer:            {tokenizer_path}")
    print(f"  aiperf --isl-block-size: 不需要 (text_input 模式)")
    print("=" * 70)

    estimated_size_gb = total_requests * 650 / (1024 ** 3)
    print(f"\n  预估文件大小: ~{estimated_size_gb:.2f} GB")

    # ---- Step 1: 加载 tokenizer ----
    print(f"\n  [1/3] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"        vocab_size = {vocab_size:,}")

    SAFE_LOW, SAFE_HIGH = 100, vocab_size

    # ---- Step 2: 生成 2M 条唯一的前缀文本 ----
    print(f"\n  [2/3] 生成 {num_unique_prompts:,} 条唯一前缀文本 ({prefix_length} tokens each)...")
    prefix_texts = []
    decode_batch = 50_000
    t0_prefix = time.time()

    for start in range(0, num_unique_prompts, decode_batch):
        end = min(start + decode_batch, num_unique_prompts)
        bs = end - start
        tokens = token_rng.randint(SAFE_LOW, SAFE_HIGH, size=(bs, prefix_length)).tolist()
        texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        prefix_texts.extend(texts)
        elapsed = time.time() - t0_prefix
        speed = end / elapsed if elapsed > 0 else 0
        eta = (num_unique_prompts - end) / speed if speed > 0 else 0
        print(f"\r        {end:>10,} / {num_unique_prompts:,}  |  "
              f"{speed:,.0f} 条/s  |  ETA: {eta:.0f}s", end="", flush=True)

    prefix_time = time.time() - t0_prefix
    print(f"\n        完成! 耗时 {prefix_time:.1f}s")

    # 验证前缀多样性 (采样)
    sample_n = min(100_000, num_unique_prompts)
    sample_idx = token_rng.choice(num_unique_prompts, sample_n, replace=False)
    sample_set = set(prefix_texts[i] for i in sample_idx)
    dup = sample_n - len(sample_set)
    if dup == 0:
        print(f"        多样性验证: {sample_n:,} 采样全部唯一 ✓")
    else:
        print(f"        多样性验证: {sample_n:,} 采样中 {dup} 个重复 (比例 {dup/sample_n:.4%})")

    # ---- Step 3: 生成 trace ----
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(
        output_path,
        f"trace_textinput_gpu{gpu_num}_qps{total_qps}_isl{isl}_osl{osl}_{duration_hours}h.jsonl",
    )
    print(f"\n  [3/3] 生成 trace: {output_file}")
    print(f"        总请求数: {total_requests:,}\n")

    chunk_size = 500_000
    num_chunks = (total_requests + chunk_size - 1) // chunk_size
    current_ts = 0.0
    written = 0
    t0 = time.time()

    with open(output_file, "w", encoding="utf-8") as f:
        for ci in range(num_chunks):
            bs = min(chunk_size, total_requests - ci * chunk_size)

            # Poisson 时间戳
            intervals = np.random.exponential(mean_interval_ms, bs)
            timestamps = np.cumsum(intervals) + current_ts
            current_ts = timestamps[-1]

            # 随机选择前缀
            prefix_indices = np.random.randint(0, num_unique_prompts, bs)

            # 生成并 decode 后缀 tokens (每请求独立)
            suffix_tokens = token_rng.randint(SAFE_LOW, SAFE_HIGH, size=(bs, suffix_length)).tolist()
            suffix_texts = tokenizer.batch_decode(suffix_tokens, skip_special_tokens=True)

            # 拼接写入
            lines = []
            for i in range(bs):
                text = prefix_texts[prefix_indices[i]] + suffix_texts[i]
                lines.append(json.dumps({
                    "timestamp": int(timestamps[i]),
                    "text_input": text,
                    "output_length": osl,
                }, separators=(",", ":"), ensure_ascii=False))

            f.write("\n".join(lines))
            if ci < num_chunks - 1:
                f.write("\n")

            written += bs
            _print_progress(written, total_requests, t0)

    total_time = time.time() - t0
    file_size = os.path.getsize(output_file)
    file_size_gb = file_size / (1024 ** 3)

    print(f"\n\n{'=' * 70}")
    print(f"  生成完成!")
    print(f"{'=' * 70}")
    print(f"  输出文件:   {output_file}")
    print(f"  文件大小:   {file_size_gb:.2f} GB ({file_size:,} bytes)")
    print(f"  总请求数:   {written:,}")
    print(f"  时间戳范围: 0 ~ {int(current_ts):,} ms ({current_ts / 1000 / 3600:.2f}h)")
    print(f"  总耗时:     {total_time + prefix_time:.1f}s (前缀生成 {prefix_time:.1f}s + trace {total_time:.1f}s)")
    print(f"")
    print(f"  数据保证:")
    print(f"    前 {prefix_length} token: 从 {num_unique_prompts:,} 条预生成文本中选取")
    print(f"      → 由 tokenizer 从 vocab 随机生成, 保证 {num_unique_prompts:,} 种不同内容")
    print(f"    后 {suffix_length} token:  每请求独立生成并 decode")
    print(f"      → 每请求不同的 {suffix_length} 个随机 token → 不同后缀文本")
    print(f"")
    print(f"  aiperf 使用示例:")
    print(f"    aiperf profile \\")
    print(f"      --model google/gemma-3-4b-it \\")
    print(f"      --tokenizer google/gemma-3-4b-it \\")
    print(f"      --endpoint-type chat \\")
    print(f"      --streaming \\")
    print(f"      --url http://localhost:8000 \\")
    print(f"      --input-file {output_file} \\")
    print(f"      --custom-dataset-type mooncake_trace \\")
    print(f"      --fixed-schedule")
    print(f"{'=' * 70}")

    return output_file


# ============================================================================
#  Verify
# ============================================================================

def verify_trace(output_path: str, num_samples: int = 5):
    """自动检测模式并验证 trace 数据"""
    import glob

    trace_files = sorted(glob.glob(os.path.join(output_path, "trace_*.jsonl")))
    if not trace_files:
        print(f"未找到 trace_*.jsonl: {output_path}")
        return

    for trace_file in trace_files:
        first_line = open(trace_file).readline().strip()
        if not first_line:
            continue
        record = json.loads(first_line)

        if "text_input" in record:
            _verify_text_input(trace_file, num_samples)
        elif "hash_ids" in record:
            _verify_hash_ids(trace_file, num_samples)
        else:
            print(f"  未知格式: {trace_file}")


def _verify_hash_ids(trace_file: str, num_samples: int):
    """验证 hash_ids 模式的 trace"""
    print(f"\n{'=' * 70}")
    print(f"  验证 [hash_ids 模式]: {os.path.basename(trace_file)}")
    print(f"{'=' * 70}")

    prefix_ids_seen = set()
    suffix_ids_seen = set()
    total = 0

    with open(trace_file, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line.strip())
            total += 1
            hash_ids = record.get("hash_ids", [])

            if len(hash_ids) == 2:
                prefix_ids_seen.add(hash_ids[0])
                suffix_ids_seen.add(hash_ids[1])

            if i < num_samples:
                print(f"\n  #{i}: ts={record['timestamp']}ms  isl={record['input_length']}  "
                      f"osl={record['output_length']}  hash_ids={hash_ids}")

    print(f"\n  总请求: {total:,}")
    print(f"  唯一 prefix_id: {len(prefix_ids_seen):,}")
    print(f"  唯一 suffix_id: {len(suffix_ids_seen):,}  "
          f"({'✓ 全部唯一' if len(suffix_ids_seen) == total else '✗ 有重复'})")

    overlap = prefix_ids_seen & suffix_ids_seen
    print(f"  ID 空间隔离:    {'✓ 无重叠' if not overlap else f'✗ {len(overlap)} 个重叠'}")

    # aiperf 兼容性
    sample = json.loads(open(trace_file).readline().strip())
    block_size = 150
    fb = sample["input_length"] - ((len(sample["hash_ids"]) - 1) * block_size)
    ok = 0 < fb <= block_size
    print(f"  aiperf 检查 (block_size={block_size}): final_block={fb}  {'✓' if ok else '✗'}")
    print(f"{'=' * 70}")


def _verify_text_input(trace_file: str, num_samples: int):
    """验证 text_input 模式的 trace"""
    print(f"\n{'=' * 70}")
    print(f"  验证 [text_input 模式]: {os.path.basename(trace_file)}")
    print(f"{'=' * 70}")

    total = 0
    text_lengths = []
    prefix_sample = {}  # 前 100 chars → count

    with open(trace_file, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line.strip())
            total += 1
            text = record.get("text_input", "")
            text_lengths.append(len(text))

            # 用前 200 字符做前缀指纹 (近似检测共享前缀)
            prefix_key = text[:200]
            prefix_sample[prefix_key] = prefix_sample.get(prefix_key, 0) + 1

            if i < num_samples:
                display = text[:80] + "..." if len(text) > 80 else text
                print(f"\n  #{i}: ts={record['timestamp']}ms  osl={record['output_length']}  "
                      f"len={len(text)} chars")
                print(f"       text: {display}")

            if total >= 1_000_000:
                break

    text_lengths = np.array(text_lengths)
    print(f"\n  扫描请求数: {total:,}")
    print(f"  text 长度:   avg={text_lengths.mean():.0f}  min={text_lengths.min()}  "
          f"max={text_lengths.max()} chars")
    print(f"  唯一前缀 (前200字符): {len(prefix_sample):,} / {total:,}")

    # 共享前缀分布
    reuse_counts = sorted(prefix_sample.values(), reverse=True)
    if len(reuse_counts) > 1:
        print(f"  前缀复用 Top-5: {reuse_counts[:5]}")
        avg_reuse = sum(reuse_counts) / len(reuse_counts)
        print(f"  平均复用次数: {avg_reuse:.1f}")

    print(f"{'=' * 70}")


# ============================================================================
#  Helpers
# ============================================================================

def _print_config(mode, gpu_num, qps_per_gpu, total_qps, duration_hours,
                  duration_seconds, total_requests, num_unique_prompts, isl,
                  prefix_length, suffix_length, osl, mean_interval_ms, seed):
    print("=" * 70)
    print(f"  Mooncake Trace 数据生成器  [mode: {mode}]")
    print("=" * 70)
    print(f"  GPU 数量:             {gpu_num}")
    print(f"  单卡 QPS:             {qps_per_gpu}")
    print(f"  总 QPS:               {total_qps:,}")
    print(f"  压测时长:             {duration_hours}h ({duration_seconds:.0f}s)")
    print(f"  总请求数:             {total_requests:,}")
    print(f"  唯一 Prefix 数:       {num_unique_prompts:,}")
    print(f"  ISL (输入长度):       {isl} (prefix={prefix_length} + suffix={suffix_length})")
    print(f"  OSL (输出长度):       {osl}")
    print(f"  平均到达间隔:         {mean_interval_ms:.6f} ms")
    print(f"  随机种子:             {seed}")
    print(f"  每个 Prefix 平均命中: {total_requests / num_unique_prompts:.1f} 次")


def _print_progress(written, total, t0):
    elapsed = time.time() - t0
    speed = written / elapsed if elapsed > 0 else 0
    progress = written / total * 100
    eta = (total - written) / speed if speed > 0 else 0
    print(f"\r  进度: {progress:6.2f}%  |  "
          f"已写入: {written:>14,} / {total:,}  |  "
          f"速度: {speed:,.0f} 条/s  |  "
          f"ETA: {eta:.0f}s", end="", flush=True)


def _print_done_hash_ids(output_file, written, current_ts, t0,
                         prefix_length, suffix_length, num_unique_prompts):
    total_time = time.time() - t0
    file_size = os.path.getsize(output_file)

    print(f"\n\n{'=' * 70}")
    print(f"  生成完成!")
    print(f"{'=' * 70}")
    print(f"  输出文件:   {output_file}")
    print(f"  文件大小:   {file_size / (1024**3):.2f} GB ({file_size:,} bytes)")
    print(f"  总请求数:   {written:,}")
    print(f"  时间戳范围: 0 ~ {int(current_ts):,} ms ({current_ts / 1000 / 3600:.2f}h)")
    print(f"  耗时:       {total_time:.1f}s")
    print(f"")
    print(f"  数据保证 (基于 aiperf PromptGenerator._build_token_sequence):")
    print(f"    前 {prefix_length} token: hash_ids[0] = prefix_id")
    print(f"      → 相同 prefix_id → aiperf._cache 命中 → 相同 token 内容")
    print(f"      → {num_unique_prompts:,} 个不同 prefix_id (实际内容种类受语料库限制)")
    print(f"    后 {suffix_length} token:  hash_ids[1] = suffix_id (每请求唯一)")
    print(f"      → 不同 suffix_id → aiperf._cache miss → 不同 token 内容")
    print(f"")
    print(f"  aiperf 使用示例:")
    print(f"    aiperf profile \\")
    print(f"      --model google/gemma-3-4b-it \\")
    print(f"      --tokenizer google/gemma-3-4b-it \\")
    print(f"      --endpoint-type chat --streaming \\")
    print(f"      --url http://localhost:8000 \\")
    print(f"      --input-file {output_file} \\")
    print(f"      --custom-dataset-type mooncake_trace \\")
    print(f"      --fixed-schedule \\")
    print(f"      --isl-block-size {prefix_length}")
    print(f"{'=' * 70}")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 Mooncake Trace benchmark 数据集 (适配 aiperf, 支持两种模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
两种模式对比:
  ┌────────────┬──────────────────────────┬──────────────────────────────┐
  │            │  hash_ids (默认)         │  text_input                  │
  ├────────────┼──────────────────────────┼──────────────────────────────┤
  │ 生成速度   │  快 (~分钟级)            │  慢 (~十分钟级, 需decode)    │
  │ 文件大小   │  小 (~15 GB/8卡2h)       │  大 (~100 GB/8卡2h)          │
  │ 前缀多样性 │  ~20-30万种 (语料库限制) │  真正200万种 (tokenizer生成)  │
  │ 后缀随机性 │  ✓ (唯一suffix_id)       │  ✓ (每请求decode新token)      │
  │ 需要       │  --isl-block-size 150    │  --tokenizer_path            │
  └────────────┴──────────────────────────┴──────────────────────────────┘

示例:
  # Mode A: hash_ids
  python generate_trace_data.py --mode hash_ids --gpu_num 8

  # Mode B: text_input
  python generate_trace_data.py --mode text_input --gpu_num 8 \\
    --tokenizer_path /raid/model/gemma-3-4b-it

  # 验证
  python generate_trace_data.py --verify
        """,
    )

    parser.add_argument(
        "--mode", choices=["hash_ids", "text_input"], default="hash_ids",
        help="生成模式 (默认: hash_ids)",
    )
    parser.add_argument(
        "--gpu_num", type=int, default=0,
        help="GPU 数量 (总 QPS = gpu_num * qps_per_gpu)",
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/raid/wenpengw/myworkspace/FlexKV/temutest",
        help="输出文件目录",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="tokenizer 路径 (text_input 模式必需, 如 /raid/model/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--qps_per_gpu", type=int, default=3000,
        help="单卡 QPS (默认: 3000)",
    )
    parser.add_argument(
        "--duration_hours", type=float, default=2.0,
        help="压测时间 (小时, 默认: 2.0)",
    )
    parser.add_argument(
        "--num_unique_prompts", type=int, default=2_000_000,
        help="不同 prompt 前缀数量 (默认: 2000000)",
    )
    parser.add_argument(
        "--prefix_length", type=int, default=150,
        help="前缀长度 (默认: 150)",
    )
    parser.add_argument(
        "--suffix_length", type=int, default=10,
        help="后缀长度 (默认: 10)",
    )
    parser.add_argument(
        "--osl", type=int, default=1,
        help="输出 token 数 (默认: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="验证已生成的数据",
    )

    args = parser.parse_args()

    if args.verify:
        verify_trace(args.output_path)
        return

    if args.gpu_num <= 0:
        print("错误: --gpu_num 必须 > 0 (或用 --verify 验证)")
        sys.exit(1)

    common_kwargs = dict(
        gpu_num=args.gpu_num,
        output_path=args.output_path,
        qps_per_gpu=args.qps_per_gpu,
        duration_hours=args.duration_hours,
        num_unique_prompts=args.num_unique_prompts,
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        osl=args.osl,
        seed=args.seed,
    )

    if args.mode == "hash_ids":
        generate_hash_ids_mode(**common_kwargs)
    elif args.mode == "text_input":
        if not args.tokenizer_path:
            print("错误: text_input 模式需要 --tokenizer_path")
            print("示例: --tokenizer_path /raid/model/gemma-3-4b-it")
            sys.exit(1)
        generate_text_input_mode(tokenizer_path=args.tokenizer_path, **common_kwargs)


if __name__ == "__main__":
    main()
