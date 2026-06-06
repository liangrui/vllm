# vLLM 指南 5：模型配置与量化优化

本指南介绍 vLLM 的量化技术及其对性能的影响。

## 5.1 量化概述

量化（Quantization）是通过降低权重精度来减少显存占用和加速推理的技术。

### 量化原理

| 精度类型 | 存储位数 | 显存占用 | 计算速度 | 精度损失 |
|---------|---------|---------|---------|---------|
| FP32 | 32bit | 1x | 1x | 无 |
| FP16/BF16 | 16bit | 0.5x | ~1x | 极低 |
| FP8 | 8bit | 0.25x | ~1.5x | 极低 |
| INT8 | 8bit | 0.25x | ~2x | 低 |
| INT4 | 4bit | 0.125x | ~4x | 中等 |

### 量化类型对比

| 量化类型 | 显存降低 | 精度损失 | 适用场景 |
|---------|---------|---------|---------|
| FP16 (无量化) | 1x | 无 | 通用 |
| FP8 | 1x | 极低 | Hopper+ GPU |
| INT8 | 2x | 低 | 所有 GPU |
| INT4 | 4x | 中等 | 显存受限 |
| GPTQ | 4x | 低 | 生产环境 |
| AWQ | 4x | 低 | 生产环境 |
| GGUF | 4x+ | 可控 | CPU/本地 |

### 量化 KV Cache

除了模型权重，KV Cache 也可以被量化：

| 类型 | 显存节省 | 质量影响 |
|------|---------|---------|
| FP16 | 0% | 无 |
| FP8_E5M2 | ~50% | 可忽略 |
| FP8_E4M3 | ~50% | 极低 |

## 5.2 自动量化

让 vLLM 自动选择最佳量化方式：

```python
from vllm import LLM

# 自动检测最优量化方式
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="auto",  # 自动检测并应用最佳量化
)
```

### 自动量化原理

1. 检测 GPU 架构
2. 评估可用显存
3. 选择最优量化方法
4. 自动应用配置

## 5.3 FP8 量化 (Hopper/Ada)

FP8 是 NVIDIA Hopper 和 Ada 架构原生支持的量化格式。

### 支持的 GPU

- H100 (Hopper)
- H200 (Hopper)
- RTX 4090 (Ada)
- RTX 3090 (Ampere) - 部分支持

### Python 配置

```python
from vllm import LLM

# FP8 W8A8 量化
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",  # FP8 量化
    dtype="half",  # 或 "float16"
)
```

### CLI 方式

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --dtype half
```

### FP8 变体

| 格式 | 描述 | 适用场景 |
|------|------|---------|
| fp8_e5m2 | 5 位指数，2 位尾数 | KV Cache |
| fp8_e4m3 | 4 位指数，3 位尾数 | 权重和激活 |

```python
# FP8 E4M3 (更高精度)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8_e4m3",
)

# FP8 E5M2 (更高范围，用于 KV Cache)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8_e5m2",
)
```

## 5.4 INT8 量化

INT8 量化使用 8 位整数存储权重，广泛兼容各种 GPU。

### BitsAndBytes 量化

```python
from vllm import LLM

# INT8 W8A8 量化
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="bitsandbytes",  # 使用 BitsAndBytes
)
```

### 量化配置选项

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="bitsandbytes",
   bnb_4bit_compute_dtype="float16",  # 计算精度
    bnb_4bit_quant_type="nf4",  # 量化类型: nf4 或 fp4
    bnb_4bit_use_double_quant=True,  # 双重量化
)
```

### BitsAndBytes 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| bnb_4bit_compute_dtype | float16, bf16, float32 | 计算时精度 |
| bnb_4bit_quant_type | nf4, fp4 | NF4 更适合神经网络 |
| bnb_4bit_use_double_quant | True, False | 减少显存占用 |

## 5.5 INT4 / AWQ / GPTQ 量化

使用预量化模型是最简单的方式。

### AWQ 量化模型

```python
from vllm import LLM

# 使用 AWQ 预量化模型
llm = LLM(
    model="lmstudio-community/Llama-3-8B-Instruct-GGUF",  # AWQ 格式
)
```

### GPTQ 量化模型

```python
from vllm import LLM

# 使用 GPTQ 预量化模型
llm = LLM(
    model="TheBloke/Llama-3-8B-Instruct-GPTQ",
)
```

### 量化模型来源

| 来源 | 量化方法 | 模型数量 |
|------|---------|---------|
| HuggingFace | AWQ, GPTQ | 大量 |
| TheBloke (GGUF) | GGUF | 最多 |
| LM Studio | GGUF, AWQ | 大量 |

### 选择预量化模型

```python
# HuggingFace 搜索
# 关键词: "AWQ", "GPTQ", "quantized"

# 示例模型
models = [
    # AWQ
    "casperhansen/llama-3-8b-instruct-awq",
    "maldicon/llama-3-8b-webglm-gcg-awq",
    
    # GPTQ
    "TheBloke/Llama-3-8B-Instruct-GPTQ",
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
]
```

## 5.6 GGUF 模型

GGUF 是高效的本地模型格式，支持 CPU 和 GPU 推理。

### GGUF 格式优势

- 单文件部署
- 支持 CPU 推理
- 多种精度选择
- 快速加载

### 使用 GGUF 模型

```python
from vllm import LLM

llm = LLM(
    model="./models/llama-3-8b.Q4_K_M.gguf",  # GGUF 文件路径
    tokenizer="./models/",  # tokenizer 路径
    gpu_memory_utilization=0.7,  # GPU 显存使用比例
)
```

### GGUF 精度对照表

| 精度 | 后缀 | 显存需求 | 质量 |
|------|------|---------|------|
| Q2_K | .q2_k | ~3GB | 较低 |
| Q3_K_M | .q3_k_m | ~4GB | 中等 |
| Q4_K_M | .q4_k_m | ~5GB | 良好 |
| Q5_K_M | .q5_k_m | ~6GB | 很好 |
| Q6_K | .q6_k | ~7GB | 接近原形 |
| Q8_0 | .q8_0 | ~8GB | 完整精度 |

### 本地 GGUF 推理

```bash
# 下载模型
# 例如: https://huggingface.co/QuantFactory/Llama-3-8b-Instruct-GGUF

# 启动服务
vllm serve ./models/Llama-3-8b-Instruct.Q4_K_M.gguf \
    --tokenizer ./models/ \
    --gpu-memory-utilization 0.8 \
    --port 8000
```

## 5.7 量化 KV Cache

KV Cache 量化可以额外节省 50% 显存。

### 配置方法

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    kv_cache_dtype="fp8_e5m2",  # 量化 KV Cache
)
```

### KV Cache 量化类型

| 类型 | 显存节省 | 质量影响 |
|------|---------|---------|
| auto | 取决于 dtype | 默认 |
| fp8_e5m2 | ~50% | 可忽略 |
| fp8_e4m3 | ~50% | 极低 |
| int8 | ~50% | 低 |

### 组合配置示例

```python
# FP8 权重 + FP8 KV Cache
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",
    kv_cache_dtype="fp8_e5m2",
)

# INT8 权重 + FP8 KV Cache
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="bitsandbytes",
    kv_cache_dtype="fp8_e5m2",
)
```

## 5.8 配置对比

### 最高精度模式（16GB+ 显存）

适用于需要最高质量的场景：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="float16",  # 或 "bfloat16"
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)
```

**特点：**
- 完整 FP16 精度
- 最大显存占用
- 最高生成质量

### 平衡模式（12GB 显存）

平衡质量和显存：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="half",  # float16 别名
    quantization="fp8",  # FP8 量化
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
```

**特点：**
- FP8 精度
- 显存减少约 30%
- 质量损失可忽略

### 省显存模式（8GB 显存）

适用于显存受限的场景：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="half",
    quantization="int8",  # INT8 量化
    max_model_len=2048,  # 限制上下文长度
    gpu_memory_utilization=0.8,
)
```

**特点：**
- INT8 精度
- 显存减少约 50%
- 适合 8GB 显卡

### 极致压缩模式（6GB 显存）

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="half",
    quantization="int8",
    max_model_len=1024,  # 进一步限制
    gpu_memory_utilization=0.7,
    kv_cache_dtype="fp8",
)
```

### 配置选择指南

| 显存 | 推荐配置 | 量化 | 最大上下文 |
|------|---------|------|-----------|
| 24GB+ | FP16 | 无 | 8192+ |
| 16GB | FP16 | 可选 | 4096-8192 |
| 12GB | FP8/INT8 | 推荐 | 2048-4096 |
| 8GB | INT8 | 必须 | 1024-2048 |
| 6GB | INT4/GGUF | 必须 | 512-1024 |

## 5.9 模型选择指南

### 按任务类型选择

| 任务 | 推荐模型 | 显存需求 | 量化方式 |
|------|---------|---------|---------|
| 快速实验 | Qwen2.5-0.5B/1.5B | 1-3GB | 无需量化 |
| 日常对话 | Qwen2.5-3B/7B | 6-16GB | FP8/INT8 |
| 代码生成 | DeepSeek-Coder-7B | 14-16GB | FP8 |
| 长上下文 | Yi-34B | 68GB | INT4 + TP |
| 高质量生成 | Llama-3.1-70B | 140GB | INT4 + TP+PP |

### 小模型推荐（<3B）

```python
# Qwen2.5-0.5B - 极致轻量
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="half",
    gpu_memory_utilization=0.9,
)

# Qwen2.5-1.5B - 平衡选择
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    dtype="half",
    gpu_memory_utilization=0.9,
)
```

### 中等模型推荐（7B）

```python
# Llama-3.1-8B - 通用
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",
    dtype="half",
    gpu_memory_utilization=0.85,
)

# Qwen2.5-7B - 中文优化
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="fp8",
    dtype="half",
    gpu_memory_utilization=0.85,
)
```

### 大模型推荐（70B+）

```python
# Llama-3.1-70B - 需要多卡
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=2,  # 2 卡
    quantization="int4_awq",
    dtype="half",
)

# Yi-34B - 长上下文
llm = LLM(
    model="01-ai/Yi-34B-Instruct",
    tensor_parallel_size=2,
    quantization="int4_awq",
    max_model_len=32768,
)
```

### 量化方法选择决策树

```
开始
  │
  ├─ GPU 是 Hopper/Ada?
  │    ├─ 是 → 使用 FP8
  │    └─ 否 → 继续
  │
  ├─ 显存 >= 16GB?
  │    ├─ 是 → 可选 INT8
  │    └─ 否 → 继续
  │
  ├─ 显存 >= 8GB?
  │    ├─ 是 → 推荐 INT8
  │    └─ 否 → 继续
  │
  └─ 使用 INT4 或 GGUF
```

### 显存计算公式

```
实际显存 ≈ (模型参数量 × 量化字节数) + KV Cache

示例（Llama-3.1-8B）:
- FP16: 8B × 2 bytes = 16GB
- FP8:  8B × 1 bytes = 8GB
- INT8: 8B × 1 bytes = 8GB
- INT4: 8B × 0.5 bytes = 4GB

KV Cache: batch_size × seq_len × 2 × hidden_size × bytes_per_param
```

### 性能基准参考

| 模型 | 量化 | 吞吐 (tokens/s) | 显存 (GB) |
|------|------|-----------------|-----------|
| Llama-3-8B | FP16 | 50 | 16 |
| Llama-3-8B | FP8 | 70 | 9 |
| Llama-3-8B | INT8 | 65 | 8 |
| Llama-3-8B | INT4 | 90 | 5 |
| Qwen2.5-7B | FP16 | 45 | 14 |
| Qwen2.5-7B | FP8 | 60 | 8 |
