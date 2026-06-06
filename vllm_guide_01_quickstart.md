# vLLM 快速安装与首轮运行

本文档介绍 vLLM 的安装配置和首次运行流程，帮助你快速上手 vLLM。

## 1.1 环境准备

在安装 vLLM 之前，请确保你的环境满足以下要求。

### 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04/22.04）
- **Python 版本**：3.10、3.11、3.12 或 3.13
- **可用磁盘空间**：至少 20GB（用于模型缓存和运行时）

### GPU 要求

- **NVIDIA GPU**：CUDA 计算能力 7.0 及以上（Volta 及更新架构）
  - 推荐计算能力 8.0 及以上（如 A100、A10G、L40S、H100 等）
  - 支持 RTX 3090、RTX 4090 等消费级 GPU
- **AMD GPU**：ROCm 6.0 及以上
- **Intel GPU**：Intel Data Center GPU Max 系列
- **Google TPU**：通过 vLLM-TPU 支持
- **CPU**：仅用于推理，功能完整但速度较慢

### 显存建议

| 模型规模 | 最低显存 | 推荐显存 |
|----------|----------|----------|
| 1-3B 参数 | 6GB | 8GB |
| 7B 参数 | 12GB | 16GB |
| 13B 参数 | 20GB | 24GB |
| 30B 参数 | 40GB | 48GB |
| 70B 参数 | 80GB | 96GB+ |

> **提示**：如果显存不足，可以考虑使用量化模型（如 AWQ、GPTQ）或减小 `max_model_len` 参数。

## 1.2 安装方式

vLLM 提供多种安装方式，从简单到灵活，任你选择。

### 方式一：uv（推荐）

uv 是新一代 Python 包管理器，速度极快，适合追求效率的用户。

```bash
# 创建虚拟环境
uv venv --python 3.12 --seed
source .venv/bin/activate

# 安装 vLLM（自动选择最优 PyTorch 后端）
uv pip install vllm --torch-backend=auto
```

> **注意**：`--torch-backend=auto` 会根据你的 CUDA 版本自动选择 PyTorch 后端。

### 方式二：pip

使用 pip 直接安装，适合快速尝鲜。

```bash
pip install vllm
```

如果你需要指定 PyTorch 版本：

```bash
# 安装 PyTorch 2.x（CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 再安装 vLLM
pip install vllm
```

### 方式三：conda

使用 conda 创建独立环境，避免依赖冲突。

```bash
# 创建 conda 环境
conda create -n vllm python=3.12 -y

# 激活环境
conda activate vllm

# 安装 vLLM
pip install vllm
```

### 方式四：Docker（最简单）

Docker 方式省去所有环境配置困扰，开箱即用。

```bash
# NVIDIA GPU 用户
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

参数说明：
- `--gpus all`：启用所有 GPU
- `-v ~/.cache/huggingface:/root/.cache/huggingface`：共享模型缓存
- `-p 8000:8000`：映射端口
- `--ipc=host`：启用共享内存，提升多 GPU 性能

## 1.3 验证安装

安装完成后，通过以下方式验证 vLLM 是否正常工作。

### Python 验证

```python
import vllm

# 查看版本
print(f"vLLM 版本: {vllm.__version__}")

# 检查 CUDA 可用性
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
```

### 命令行验证

```bash
# 检查 vLLM CLI 是否可用
vllm --version

# 列出可用模型（需要先下载模型）
vllm list
```

## 1.4 首轮运行

完成安装后，让我们运行第一个 vLLM 程序。

### 基础示例

```python
from vllm import LLM, SamplingParams

# 第一步：初始化模型
# vLLM 会自动下载模型到 ~/.cache/huggingface/
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 第二步：定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,      # 控制随机性，0 为贪婪解码
    top_p=0.9,           # 核采样阈值
    max_tokens=100,      # 最大生成 100 个 token
)

# 第三步：生成文本
output = llm.generate("你好，请介绍一下你自己", sampling_params)

# 第四步：输出结果
print(output[0].outputs[0].text)
```

### 参数说明

- `temperature`：控制输出的随机性
  - `0.0`：贪婪解码，输出确定
  - `0.7`：平衡模式（推荐）
  - `1.0+`：高随机性，适合创意任务
- `top_p`：核采样参数，保留概率和超过此值的 token
- `max_tokens`：允许生成的最大 token 数

## 1.5 常见问题排查

以下是常见问题及解决方案。

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **CUDA 版本不兼容** | PyTorch 与 CUDA 版本不匹配 | 使用 `uv pip install vllm --torch-backend=auto`，或手动安装匹配版本的 PyTorch |
| **显存不足 (OOM)** | 模型太大或 `gpu_memory_utilization` 太高 | 减小 `max_model_len`、降低 `gpu_memory_utilization` 到 0.7 或更低、使用更小的模型 |
| **模型下载失败** | 网络问题或 HuggingFace 访问受限 | 设置 `VLLM_USE_MODELSCOPE=True` 使用 ModelScope，或配置代理 |
| **import 失败** | 缺少依赖 | 重新安装 vLLM 或检查 PyTorch 安装 |
| **GPU 未识别** | CUDA 驱动问题 | 确认已安装 NVIDIA 驱动，运行 `nvidia-smi` 检查 |

### 网络问题解决方案

如果下载模型时遇到网络问题，可以尝试：

```bash
# 使用 ModelScope 代替 HuggingFace
export VLLM_USE_MODELSCOPE=True

# 或在代码中设置
import os
os.environ["VLLM_USE_MODELSCOPE"] = "True"
```

### 显存优化技巧

```python
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    gpu_memory_utilization=0.7,  # 降低到 70%
    max_model_len=2048,          # 限制上下文长度
    tensor_parallel_size=1,      # 单卡
)
```

## 1.6 支持的硬件

vLLM 支持多种硬件平台，满足不同部署需求。

| 硬件 | 安装方式 | 备注 |
|------|----------|------|
| **NVIDIA GPU (CUDA)** | `pip install vllm` | 最常用，性能最优 |
| **AMD GPU (ROCm)** | `--extra-index-url https://wheels.vllm.ai/rocm/` | 需要 ROCm 6.0+ |
| **Intel GPU (XPU)** | `pip install vllm` | 需要 IPEX 驱动 |
| **Google TPU** | `pip install vllm-tpu` | 通过 vllm-tpu 包支持 |
| **CPU** | `pip install vllm` | 仅用于测试，生产不推荐 |

### NVIDIA GPU 额外安装

```bash
# 确认 CUDA 版本
nvcc --version

# 推荐使用 auto 后端自动匹配
uv pip install vllm --torch-backend=auto
```

### AMD GPU 安装

```bash
# 添加 ROCm 索引
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

## 1.7 下一步

安装完成后，你可以继续学习：

- **离线推理基础**：[vllm_guide_02_offline_inference.md](./vllm_guide_02_offline_inference.md) — 了解批量推理、SamplingParams 详解、输出结构等
- **在线服务部署**：[vllm_guide_03_online_serving.md](./vllm_guide_03_online_serving.md) — 了解如何部署 API 服务器、调用 API、配置认证等

---

祝你玩得开心！如果遇到问题，欢迎查阅 vLLM 官方文档或提交 Issue。
