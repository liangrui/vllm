# vLLM 离线推理基础

本文档介绍如何使用 vLLM 进行离线推理，适合批量处理、实验研究和算法验证场景。

## 2.1 LLM 类概述

`LLM` 类是 vLLM 离线推理的核心接口，负责模型的加载和推理执行。

### 基本用法

```python
from vllm import LLM, SamplingParams

# 初始化 LLM
llm = LLM(model="facebook/opt-125m")

# 定义采样参数
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

# 推理
output = llm.generate("The capital of France is", sampling_params)
print(output[0].outputs[0].text)
```

### 常用参数

```python
llm = LLM(
    # 模型配置
    model="facebook/opt-125m",       # HuggingFace 模型名或本地路径
    tokenizer="facebook/opt-125m",   # 指定 tokenizer（可选）
    tokenizer_mode="auto",            # auto、slow、mistral
    
    # 并行配置
    tensor_parallel_size=1,           # 张量并行大小（多 GPU）
    pipeline_parallel_size=1,        # 流水线并行大小
    
    # 显存配置
    gpu_memory_utilization=0.9,      # GPU 显存使用比例（0.0-1.0）
    max_model_len=2048,              # 最大模型长度（序列长度）
    
    # 数据类型
    dtype="auto",                    # auto、float16、bfloat16、float32
    
    # 量化配置
    quantization="awq",              # 量化方法：awq、gptq、fp8 等
    
    # 其他
    trust_remote_code=False,         # 是否信任远程代码
    seed=None,                       # 随机种子
)
```

### 参数详解

#### model

模型标识符，可以是：
- HuggingFace 模型名：如 `"Qwen/Qwen2.5-1.5B-Instruct"`
- 本地路径：如 `"./models/my-model"`
- ModelScope 模型名（需设置 `VLLM_USE_MODELSCOPE=True`）

#### tensor_parallel_size

张量并行大小，用于多 GPU 场景。将模型切分到多个 GPU：

```python
# 使用 2 张 GPU
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
)
```

#### gpu_memory_utilization

控制 vLLM 使用的 GPU 显存比例。默认 0.9，即使用 90% 显存：

```python
# 保守设置，保留更多显存
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.7,
)
```

#### max_model_len

最大序列长度，包括 prompt 和生成的 tokens。增大此值会消耗更多显存：

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=4096,  # 支持 4096 长度的上下文
)
```

## 2.2 批量推理

vLLM 擅长批量处理多个请求，显著提升吞吐量。

### 基本批量推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# 准备多个 prompts
prompts = [
    "The capital of France is",
    "The largest mammal is",
    "Python is a",
    "Machine learning is",
    "The sun rises in the",
]

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

# 批量生成
outputs = llm.generate(prompts, sampling_params)

# 遍历结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("---")
```

### 批量大小的影响

| 批量大小 | 适用场景 | 注意事项 |
|----------|----------|----------|
| 1 | 单请求、测试 | 最简单 |
| 8-32 | 日常推理 | 平衡吞吐和延迟 |
| 64+ | 批处理任务 | 高吞吐，延迟较高 |

## 2.3 SamplingParams 详解

`SamplingParams` 控制生成过程的所有方面。

### 完整参数列表

```python
sampling_params = SamplingParams(
    # ========== 随机性控制 ==========
    temperature=0.7,      # 温度参数
                         # 0.0 = 贪婪解码（确定性输出）
                         # 0.7 = 平衡模式（推荐默认值）
                         # 1.0+ = 高随机性（创意写作）
    
    top_p=0.9,           # 核采样阈值
                         # 保留概率和超过此值的最小 token 集合
                         # 1.0 = 禁用核采样
                         # 0.9 = 保留概率和为 0.9 的 token（推荐）
    
    top_k=50,            # Top-K 采样
                         # 只保留概率最高的 k 个 token
                         # -1 = 禁用 top-k
    
    min_p=0.1,           # 最小概率阈值
                         # 忽略概率低于 max(prob * min_p, top_k_token_prob) 的 token
    
    # ========== 长度控制 ==========
    max_tokens=100,      # 最大生成 token 数
                        # 控制输出长度上限
    
    min_tokens=0,       # 最小生成 token 数
                        # 确保输出至少这么多 token（慎用，可能提前截断）
    
    # ========== 重复控制 ==========
    repetition_penalty=1.0,  # 重复惩罚
                            # 1.0 = 无惩罚
                            # 1.1 = 轻度惩罚（推荐）
                            # 1.2+ = 强力惩罚
    
    length_penalty=1.0,  # 长度惩罚（用于 beam search）
    
    # ========== 停止条件 ==========
    stop=None,          # 停止词列表
                        # 遇到这些字符串时停止生成
                        # stop=["\n", "END"]
    
    stop_token_ids=None, # 停止 token ID 列表
                         # 遇到这些 token 时停止生成
    
    include_stop_str_in_output=False,  # 输出中是否包含停止词
    
    # ========== Logprobs ==========
    logprobs=None,      # 返回 top-k logprobs
                        # None = 不返回
                        # 3 = 返回 top-3
    
    prompt_logprobs=None, # 返回 prompt token 的 logprobs
    
    # ========== Beam Search ==========
    n=1,                # 生成候选数量
                        # 1 = 单输出
                        # 3 = 生成 3 个候选供选择
    
    best_of=1,          # beam 宽度（用于 beam search）
    
    use_beam_search=False,  # 是否使用 beam search
    
    # ========== 其他 ==========
    seed=None,          # 随机种子（需同时设置 llm 的 seed）
    
    skip_special_tokens=True,  # 是否跳过特殊 token
    
    spaces_between_special_tokens=True,  # 特殊 token 之间是否加空格
)
```

### 常见场景配置

#### 贪婪解码（确定性输出）

```python
sampling_params = SamplingParams(
    temperature=0.0,    # 必须设为 0
    max_tokens=100,
)
```

#### 创意写作

```python
sampling_params = SamplingParams(
    temperature=1.0,    # 高随机性
    top_p=0.95,         # 配合高 top_p
    top_k=50,
    max_tokens=500,
    repetition_penalty=1.1,
)
```

#### 精确问答

```python
sampling_params = SamplingParams(
    temperature=0.3,    # 低随机性
    top_p=0.9,
    top_k=20,
    max_tokens=200,
    repetition_penalty=1.05,
)
```

## 2.4 输出结构

理解 vLLM 的输出结构，便于后续处理。

### 完整输出结构

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    # ========== 请求级别信息 ==========
    print(f"Request ID: {output.request_id}")        # 请求唯一 ID
    print(f"Prompt: {output.prompt}")                  # 输入 prompt
    print(f"Prompt Token IDs: {output.prompt_token_ids}")  # Prompt 的 token ID 列表
    print(f"Prompt Logprobs: {output.prompt_logprobs}")    # Prompt 每个位置的 logprob
    
    # ========== 候选输出（可能有多个） ==========
    for i, generated_output in enumerate(output.outputs):
        print(f"\n--- Output {i} ---")
        print(f"  Text: {generated_output.text}")          # 生成的文本
        print(f"  Token IDs: {generated_output.token_ids}") # 生成的 token ID
        print(f"  Logprobs: {generated_output.logprobs}")  # 每个位置的 logprob
        print(f"  Finish Reason: {generated_output.finish_reason}")  # 停止原因
    
    # ========== 统计信息 ==========
    prompt_len = len(output.prompt_token_ids)
    completion_len = len(output.outputs[0].token_ids)
    total_len = prompt_len + completion_len
    
    print(f"\n统计信息:")
    print(f"  Prompt Tokens: {prompt_len}")
    print(f"  Completion Tokens: {completion_len}")
    print(f"  Total Tokens: {total_len}")
```

### Finish Reason 说明

生成停止的原因：

| 值 | 含义 |
|----|------|
| `length` | 达到 `max_tokens` 限制 |
| `stop` | 遇到停止词或 `stop_token_ids` |
| `eos` | 生成了 EOS token |
| `timeout` | 达到超时限制 |

### 处理 Logprobs

```python
# 请求 logprobs
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=50,
    logprobs=5,         # 返回 top-5 logprobs
    prompt_logprobs=5,  # 同时返回 prompt logprobs
)

outputs = llm.generate(["Hello, world!"], sampling_params)

for output in outputs:
    # 生成部分的 logprobs
    for pos, logprob_info in enumerate(output.outputs[0].logprobs):
        if logprob_info:
            print(f"Position {pos}:")
            for token, logprob in logprob_info.items():
                print(f"  {token!r}: {logprob:.4f}")
    
    # Prompt 部分的 logprobs
    for pos, logprob_info in enumerate(output.prompt_logprobs):
        if logprob_info:
            print(f"Prompt Position {pos}: {logprob_info}")
```

## 2.5 Chat 界面

vLLM 提供 `llm.chat()` 接口，自动处理 chat template。

### 基本用法

```python
# 使用 chat 接口
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是大语言模型？"},
]

outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs[0].text)
```

### Chat 与 Generate 的区别

| 特性 | `llm.generate()` | `llm.chat()` |
|------|------------------|--------------|
| 输入格式 | 原始字符串 | 消息列表 |
| Chat Template | 不应用 | 自动应用 |
| 多轮对话 | 需手动拼接 | 原生支持 |
| 适用场景 |Completion 任务 | 对话任务 |

### 多轮对话

```python
messages = [
    {"role": "system", "content": "你是一个Python编程助手。"},
    {"role": "user", "content": "如何定义一个函数？"},
    {"role": "assistant", "content": "在Python中，你可以使用def关键字来定义函数。\n\n```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```"},
    {"role": "user", "content": "如何添加类型注解？"},
]

outputs = llm.chat(messages, sampling_params)
print(outputs[0].outputs[0].text)
```

### 指定 Chat Template

```python
# 自动检测（默认）
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 手动指定
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    chat_template=None,  # 使用模型自带的 template
)
```

## 2.6 高级用法

### 多次采样

生成多个候选，选择最佳结果：

```python
sampling_params = SamplingParams(
    n=3,              # 生成 3 个候选
    temperature=0.8,
    max_tokens=50,
    top_p=0.95,
)

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {output.prompt!r}")
    for j, cand in enumerate(output.outputs):
        print(f"  [{j}] {cand.text}")
    print()
```

### Beam Search

适用于需要高质量、完整句子的任务：

```python
sampling_params = SamplingParams(
    n=3,                # 返回 3 个候选
    best_of=3,          # beam 宽度
    use_beam_search=True,  # 启用 beam search
    temperature=0.0,    # 通常设为 0
    max_tokens=100,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Best: {output.outputs[0].text}")
```

### 自定义 SamplingParams

```python
# 工厂函数
def create_sampling_params(
    mode: str = "balanced",
    max_tokens: int = 100,
):
    presets = {
        "greedy": {"temperature": 0.0, "top_p": 1.0},
        "balanced": {"temperature": 0.7, "top_p": 0.9},
        "creative": {"temperature": 1.0, "top_p": 0.95, "top_k": 50},
    }
    params = presets.get(mode, presets["balanced"])
    params["max_tokens"] = max_tokens
    return SamplingParams(**params)

# 使用
sampling_params = create_sampling_params(mode="creative", max_tokens=200)
```

### 异步推理

```python
import asyncio
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

async def generate_async(prompts, sampling_params):
    # vLLM 支持异步调用
    outputs = await llm.generate(prompts, sampling_params)
    return outputs

# 运行
asyncio.run(generate_async(["Hello!"], SamplingParams(max_tokens=50)))
```

## 2.7 性能优化建议

### 提升吞吐量

1. **批量处理**：尽可能将多个请求组成批量
2. **增大 gpu_memory_utilization**：在显存允许范围内
3. **使用量化**：如 AWQ、GPTQ 量化模型

### 降低延迟

1. **使用更小的模型**：如 7B 代替 70B
2. **减小 max_model_len**：只分配需要的上下文长度
3. **贪婪解码**：temperature=0 可降低延迟

### 显存不足处理

```python
# 方案一：降低显存使用
llm = LLM(
    model="large-model",
    gpu_memory_utilization=0.5,
    max_model_len=1024,
)

# 方案二：使用量化模型
llm = LLM(
    model="large-model-AWQ",
    quantization="awq",
)

# 方案三：张量并行（多卡）
llm = LLM(
    model="large-model",
    tensor_parallel_size=2,
)
```

---

恭喜你完成离线推理基础学习！下一步可以学习 [在线服务部署](./vllm_guide_03_online_serving.md)，了解如何通过 API 提供服务。
