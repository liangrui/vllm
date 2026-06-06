# vLLM 高级指南 8：分布式部署

## 8.1 张量并行（Tensor Parallelism）

### 概述

张量并行（Tensor Parallelism, TP）将模型的单个层切分到多个 GPU 上，使得大模型可以加载到多个 GPU 的显存中。每个 GPU 持有模型参数的一部分，所有 GPU 协同完成一次前向传播。

### 工作原理

```
传统模式（单 GPU）：
┌─────────────────────────────────────┐
│            Linear Layer              │
│         [W: 70B parameters]          │
└─────────────────────────────────────┘

张量并行（4 GPU）：
┌──────────┬──────────┬──────────┬──────────┐
│  W[:,0]  │  W[:,1]  │  W[:,2]  │  W[:,3]  │
│   17.5B   │   17.5B   │   17.5B   │   17.5B   │
└──────────┴──────────┴──────────┴──────────┘
      │          │          │          │
      ▼          ▼          ▼          ▼
   [GPU 0]    [GPU 1]    [GPU 2]    [GPU 3]
```

### 基本配置

```python
from vllm import LLM

# 单节点 4 GPU，张量并行 4
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    dtype="half",
)
```

```bash
# CLI 方式
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

### 张量并行的通信模式

每个 transformer 层内：
1. **Column 并行**：线性层按列切分（Y = WX，W 按列切分）
2. **AllReduce**：汇总各 GPU 的部分结果
3. **Row 并行**：线性层按行切分（保持输出完整）

```
计算流程：
GPU 0: W0 @ X → Y0
GPU 1: W1 @ X → Y1
GPU 2: W2 @ X → Y2
GPU 3: W3 @ X → Y3
           ↓ AllReduce
      Y = Y0 + Y1 + Y2 + Y3
```

## 8.2 流水线并行（Pipeline Parallelism）

### 概述

流水线并行（Pipeline Parallelism, PP）将模型的不同层分配到不同的 GPU 上，每个 GPU 负责模型的一部分层。适合超大规模模型的分布式部署。

### 工作原理

```
流水线并行示意（4 GPU, 32 层模型）：

GPU 0: [Layer 0-7]  →  GPU 1: [Layer 8-15]  →  GPU 2: [Layer 16-23]  →  GPU 3: [Layer 24-31]
   │                        │                        │                        │
   ▼                        ▼                        ▼                        ▼
 Micro-batch 1 ─────►  Micro-batch 1 ─────►  Micro-batch 1 ─────►  Micro-batch 1
 Micro-batch 2 ─────►  Micro-batch 2 ─────►  Micro-batch 2 ─────►  Micro-batch 2
 Micro-batch 3 ─────►  Micro-batch 3 ─────►  Micro-batch 3 ─────►  Micro-batch 3
   ...                      ...                      ...                      ...
```

### 配置方法

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,  # 总共 8 GPU
)
```

```bash
# CLI 方式
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

### 流水线并行的挑战

- **流水线气泡**：由于微批次顺序处理，GPU 有空闲时间
- **通信开销**：需要传输层间激活值
- **内存不平衡**：部分 GPU 需要存储更多激活值

### 优化策略

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    max_num_concurrent_batches=4,  # 增加并发减少气泡
)
```

## 8.3 数据并行（Data Parallelism）

### 概述

数据并行（Data Parallelism, DP）复制多个模型副本，每个 GPU 独立处理不同的请求批次。适合高吞吐场景，可以线性扩展吞吐量。

### 工作原理

```
数据并行（4 副本）：

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Replica 0  │  │  Replica 1  │  │  Replica 2  │  │  Replica 3  │
│  GPU 0      │  │  GPU 1      │  │  GPU 2      │  │  GPU 3      │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ Batch 0     │  │ Batch 1     │  │ Batch 2     │  │ Batch 3     │
│ [req0,req1] │  │ [req2,req3] │  │ [req4,req5] │  │ [req6,req7] │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### 基本配置

```bash
# 启动数据并行（4 副本）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 4
```

### 结合张量并行

```bash
# 2 路数据并行 + 2 路张量并行 = 4 GPU
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 2 \
    --tensor-parallel-size 2
```

### 数据并行的工作模式

vLLM 的数据并行采用 **worker group** 架构：
- 一个 primary worker 处理请求分发
- 多个 secondary workers 独立处理请求
- 无需模型权重同步（每个节点独立）

```bash
# Node 0 (primary)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address 10.0.0.1

# Node 1 (secondary)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address 10.0.0.1 \
    --headless
```

## 8.4 多节点部署

### 多节点架构

vLLM 支持跨多节点的数据并行，通过 RPC 通信协调各节点。

```
节点 0 (Head)
┌─────────────────────────────────────────────────┐
│ Primary Worker                                   │
│ - 请求分发                                        │
│ - 结果汇总                                        │
│ - 与节点 1-3 通信                                │
└─────────────────────────────────────────────────┘

节点 1-3 (Workers)
┌─────────────────────────────────────────────────┐
│ Secondary Worker                                 │
│ - 处理分配到的请求                                │
│ - 返回结果给节点 0                               │
└─────────────────────────────────────────────────┘
```

### 配置示例

```bash
# Node 0 (head)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345
```

```bash
# Node 1
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345
```

### 多节点配置参数

| 参数 | 说明 |
|------|------|
| `--data-parallel-size` | 总副本数 |
| `--data-parallel-size-local` | 本地副本数 |
| `--data-parallel-start-rank` | 本节点起始 rank |
| `--data-parallel-address` | primary worker 地址 |
| `--data-parallel-rpc-port` | RPC 通信端口 |

## 8.5 专家并行（Expert Parallelism）

### 概述

专家并行（Expert Parallelism, EP）是针对 MoE（Mixture of Experts）模型的特殊并行策略。MoE 模型包含多个"专家"网络，每次推理只激活部分专家。

### 工作原理

```
MoE 层结构：
┌─────────────────────────────────────────┐
│              MoE Layer                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│  │Exp 0│ │Exp 1│ │Exp 2│ │Exp 3│ ...    │
│  └─────┘ └─────┘ └─────┘ └─────┘        │
│        ↑         ↑         ↑            │
│        │ Router 选择部分专家激活          │
└─────────────────────────────────────────┘

专家并行（4 GPU）：
GPU 0: Expert 0, 4, 8, 12, ...
GPU 1: Expert 1, 5, 9, 13, ...
GPU 2: Expert 2, 6, 10, 14, ...
GPU 3: Expert 3, 7, 11, 15, ...
```

### 配置方法

```bash
# DeepSeek-V3 等 MoE 模型
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel
```

### 适用模型

- DeepSeek-V3
- Mixtral
- DBRX
- Qwen-MoE

## 8.6 上下文并行（Context Parallelism）

### 概述

上下文并行（Context Parallelism, CP）将超长上下文切分到多个 GPU，适合长上下文推理场景。

### 使用场景

- 长文档摘要
- 长程对话
- 代码补全（大型代码库）
- 多文档分析

### 配置方法

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --context-parallel-size 4  # 将上下文切分到 4 GPU
```

### 工作原理

```
上下文并行示意（4 GPU, 8192 token 上下文）：

GPU 0: Token 0-2047    GPU 1: Token 2048-4095
GPU 2: Token 4096-6143 GPU 3: Token 6144-8191

Attention 计算时：
- Query 在各 GPU 本地计算
- Key, Value 需要跨 GPU 通信
- 使用环形通信模式
```

## 8.7 并行策略选择

### 模型规模与 GPU 配置对照表

| 模型规模 | GPU 数量 | 推荐配置 | 说明 |
|---------|---------|---------|------|
| 7B | 1 | 无并行 | 单卡可加载 |
| 7B | 2 | TP=2 | 提升性能 |
| 7B | 4 | TP=4 | 最佳单模型性能 |
| 13B | 1 | 无并行 | 可能需要量化 |
| 13B | 4 | TP=4 | 高吞吐部署 |
| 34B | 4 | TP=4 | 需要量化或 TP |
| 34B | 8 | TP=4, PP=2 | 超大显存配置 |
| 70B | 4 | TP=4 | 极限配置，需要量化 |
| 70B | 8 | TP=8 | 推荐配置 |
| 70B | 16 | TP=8, PP=2 | 高可用部署 |
| 405B | 8 | TP=8 | 极限配置，需量化 |
| 405B | 16 | TP=8, PP=2 | 推荐配置 |
| 405B | 32 | TP=8, PP=4 | 超大规模部署 |

### 选型决策树

```
开始
  │
  ▼
模型参数量是多少？
  │
  ├── < 7B ──→ 单卡是否够？
  │            ├── 是 → 单卡部署
  │            └── 否 → TP=2 或 TP=4
  │
  ├── 7B-13B ─→ 有多少 GPU？
  │            ├── 1 → 量化 + 单卡
  │            └── 4+ → TP=4
  │
  ├── 13B-70B ─→ 有多少 GPU？
  │             ├── 4 → TP=4（可能需量化）
  │             └── 8+ → TP=8 或 TP=4+PP=2
  │
  └── > 70B ──→ 需要多少 GPU？
               ├── 8 → TP=8（需量化）
               ├── 16 → TP=8 + PP=2
               └── 32+ → TP=8 + PP=4
```

## 8.8 显存计算

### 显存需求估算公式

```
总显存需求 ≈ 模型参数显存 + KV Cache 显存

模型参数显存 = (模型参数量 × dtype大小) / 张量并行数
KV Cache 显存 ≈ batch_size × max_tokens × 2 × layers × hidden_size × dtype大小 / 压缩比
```

### 各 dtype 的大小

| dtype | 大小（bytes） | 说明 |
|-------|--------------|------|
| float32 | 4 | 全精度 |
| float16 / half | 2 | 半精度 |
| bfloat16 | 2 | BF16，推荐 |
| int8 | 1 | 量化 |
| int4 | 0.5 | 高压缩量化 |

### 计算示例

**示例 1: Llama-70B, FP16, TP=4**
```
模型参数显存 = (70B × 2 bytes) / 4 = 35GB
KV Cache 显存 ≈ 20GB（典型配置）
─────────────────────────────
总需求 ≈ 55GB

每 GPU 需求 = 55GB / 4 ≈ 14GB
结论：需要 4×16GB 或 2×40GB 或 1×80GB
```

**示例 2: Llama-7B, FP16, 单卡**
```
模型参数显存 = (7B × 2 bytes) / 1 = 14GB
KV Cache 显存 ≈ 8GB（典型配置）
─────────────────────────────
总需求 ≈ 22GB

结论：需要 1×24GB 或优化配置
```

**示例 3: Llama-70B, INT8, TP=4**
```
模型参数显存 = (70B × 1 bytes) / 4 = 17.5GB
KV Cache 显存 ≈ 15GB（量化后增大）
─────────────────────────────
总需求 ≈ 32.5GB

每 GPU 需求 ≈ 8GB
结论：4×12GB 即可部署
```

### 实用显存计算器

```python
def calculate_memory(model_params: int, dtype: str, tp_size: int, 
                     batch_size: int, max_tokens: int, 
                     num_layers: int, hidden_size: int) -> dict:
    """计算显存需求"""
    
    dtype_map = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    dtype_size = dtype_map.get(dtype, 2)
    
    # 模型参数显存
    model_memory = (model_params * dtype_size) / tp_size
    
    # KV Cache 估算（简化模型）
    # 每个 token 需要 2 * num_layers * hidden_size * dtype_size bytes
    kv_per_token = 2 * num_layers * hidden_size * dtype_size
    kv_cache = batch_size * max_tokens * kv_per_token / (1024**3)  # GB
    
    total = model_memory + kv_cache
    
    return {
        "model_memory_gb": model_memory,
        "kv_cache_gb": kv_cache,
        "total_gb": total,
        "per_gpu_gb": total / tp_size
    }

# 使用示例
result = calculate_memory(
    model_params=70_000_000_000,  # 70B
    dtype="float16",
    tp_size=4,
    batch_size=32,
    max_tokens=4096,
    num_layers=80,
    hidden_size=8192
)
print(f"总显存需求: {result['total_gb']:.1f} GB")
print(f"每 GPU 需求: {result['per_gpu_gb']:.1f} GB")
```

## 8.9 分布式部署最佳实践

### 网络配置

```bash
# 使用 InfiniBand 提升多节点通信
# 在 NCCL 配置中指定
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NET_GDR_LEVEL=IB

# 或使用 RoCE
export NCCL_NET_GDR_LEVEL=PHB
```

### 启动脚本模板

```bash
#!/bin/bash
# launch_distributed.sh

GPUS_PER_NODE=8
NNODES=2
NODE_RANK=0
MASTER_ADDR="10.0.0.1"
MASTER_PORT=29500

MODEL="meta-llama/Llama-3.1-70B-Instruct"
TP=8
PP=2
DP=2

export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=22

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --pipeline-parallel-size $PP \
    --data-parallel-size $DP \
    --data-parallel-size-local $((GPUS_PER_NODE / TP)) \
    --data-parallel-address $MASTER_ADDR \
    --data-parallel-rpc-port 13345 \
    --data-parallel-start-rank $((NODE_RANK * GPUS_PER_NODE / TP)) \
    --host 0.0.0.0 \
    --port 8000
```

### 性能调优

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,
    # 通信优化
    collective_ranking_mode=True,
    # 内存优化
    gpu_memory_utilization=0.92,
    # 调度优化
    max_num_concurrent_batches=4,
    prefill_chunk_size=1024,
)
```

## 8.10 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| NCCL 初始化失败 | GPU 间通信不可达 | 检查 InfiniBand/NIC 配置 |
| 显存不足 | 模型过大或 TP 不足 | 增加 TP size 或使用量化 |
| 吞吐量不随 DP 线性增长 | 负载不均衡 | 增加副本数或优化请求分发 |
| 通信成为瓶颈 | GPU 间带宽不足 | 使用更快的网络硬件 |
| 流水线气泡过多 | 微批次太少 | 增加 `max_num_concurrent_batches` |

### 调试命令

```bash
# 检查 NCCL 通信
export NCCL_DEBUG=INFO
vllm serve ...

# 检查 GPU 拓扑
nvidia-smi topo -m

# 检查 GPU 间带宽
python -c "import torch; torch.cuda.nccl.enabled(); print(torch.cuda.nccl.version())"
```

## 8.11 高可用配置

### 多副本高可用

```bash
# 副本 1
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 3 \
    --data-parallel-size-local 1 \
    --data-parallel-address 10.0.0.1 \
    --port 8000

# 副本 2
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 3 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address 10.0.0.1 \
    --port 8000

# 副本 3
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 3 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.0.0.1 \
    --port 8000
```

### 健康检查与自动恢复

```bash
# 健康检查脚本
#!/bin/bash
for i in {1..3}; do
    if curl -s http://10.0.0.$i:8000/health > /dev/null; then
        echo "Replica $i: OK"
    else
        echo "Replica $i: FAILED"
        # 触发告警或自动重启
    fi
done
```
