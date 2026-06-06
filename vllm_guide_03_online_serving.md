# vLLM 在线服务部署

本文档介绍如何使用 vLLM 部署在线 API 服务，支持 OpenAI 兼容接口。

## 3.1 启动 API 服务器

vLLM 提供开箱即用的 API 服务器，部署简单高效。

### 基础启动

```bash
# 使用 vllm serve 命令启动服务器
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

服务器启动后，默认监听 `http://localhost:8000`。

### 带参数启动

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \              # 监听所有网络接口
    --port 8000 \                 # 端口号
    --gpu-memory-utilization 0.9 \ # GPU 显存使用比例
    --max-model-len 4096 \        # 最大模型长度
    --tensor-parallel-size 2 \    # 张量并行（多 GPU）
    --quantization awq            # 量化方法
```

### 常用启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | `127.0.0.1` |
| `--port` | 监听端口 | `8000` |
| `--gpu-memory-utilization` | 显存使用比例 | `0.9` |
| `--max-model-len` | 最大序列长度 | `2048` |
| `--tensor-parallel-size` | GPU 数量 | `1` |
| `--pipeline-parallel-size` | 流水线并行 | `1` |
| `--quantization` | 量化方法 | `None` |
| `--dtype` | 数据类型 | `auto` |
| `--api-key` | API 认证密钥 | `None` |
| `--allowed-origins` | 允许的源 | `["*"]` |
| `--max-concurrent-requests` | 最大并发 | `2147483647` |

### 多 GPU 部署

```bash
# 2 GPU 张量并行
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 2

# 4 GPU 张量并行
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4
```

### Docker 部署

```bash
# 单 GPU
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    vllm serve Qwen/Qwen2.5-1.5B-Instruct

# 多 GPU
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 2
```

## 3.2 API 端点

vLLM 提供 OpenAI 兼容的 REST API。

### Completions API

用于文本补全任务。

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": false
    }'
```

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称 |
| `prompt` | string/string[] | 是 | 输入文本 |
| `max_tokens` | integer | 否 | 最大 token 数 |
| `temperature` | float | 否 | 温度参数 |
| `top_p` | float | 否 | 核采样阈值 |
| `stream` | boolean | 否 | 是否流式输出 |

**响应示例**：

```json
{
    "id": "cmpl-abc123",
    "object": "text_completion",
    "created": 1234567890,
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "choices": [
        {
            "text": "Paris, the capital of France, is known for the Eiffel Tower.",
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 12,
        "total_tokens": 17
    }
}
```

### Chat Completions API

用于对话任务。

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }'
```

**消息格式**：

```json
{
    "messages": [
        {"role": "system", "content": "系统消息（可选）"},
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "助手消息（可选，用于多轮对话）"}
    ]
}
```

**响应示例**：

```json
{
    "id": "chatcmpl-xyz789",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "法国的首都是巴黎。巴黎位于法国北部，是法国的政治、经济和文化中心。著名的埃菲尔铁塔就坐落在巴黎。"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 30,
        "completion_tokens": 50,
        "total_tokens": 80
    }
}
```

### Embeddings API

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": "The quick brown fox jumps over the lazy dog"
    }'
```

### 其他端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/model_info/{model_name}` | GET | 获取模型信息 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |

```bash
# 列出可用模型
curl http://localhost:8000/v1/models

# 健康检查
curl http://localhost:8000/health
```

## 3.3 Python 客户端

使用 OpenAI Python SDK 调用 vLLM 服务。

### 安装依赖

```bash
pip install openai
```

### 基本用法

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    api_key="EMPTY",              # 本地服务不需要 API key
    base_url="http://localhost:8000/v1",  # vLLM 服务地址
)

# ========== Chat Completions ==========
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "你好！"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(chat_response.choices[0].message.content)

# ========== Completions ==========
completion = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7,
)

print(completion.choices[0].text)

# ========== 获取使用统计 ==========
print(f"\n使用统计:")
print(f"  Prompt Tokens: {chat_response.usage.prompt_tokens}")
print(f"  Completion Tokens: {chat_response.usage.completion_tokens}")
print(f"  Total Tokens: {chat_response.usage.total_tokens}")
```

### 使用 Response 对象

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

# 获取响应 ID
print(f"Response ID: {response.id}")

# 获取模型
print(f"Model: {response.model}")

# 获取创建时间
print(f"Created: {response.created}")

# 获取完成选择
choice = response.choices[0]
print(f"Finish Reason: {choice.finish_reason}")
print(f"Message: {choice.message.content}")
print(f"Message Role: {choice.message.role}")
```

## 3.4 流式输出

vLLM 支持 Server-Sent Events (SSE) 流式输出。

### curl 流式调用

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "讲一个笑话"}],
        "stream": true
    }'
```

### Python 流式调用

```python
# 流式 Chat Completions
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "讲一个笑话"}],
    stream=True,
    max_tokens=200,
)

print("助手: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行

# 流式 Completions
stream = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="The capital of France is",
    stream=True,
    max_tokens=50,
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
print()
```

### 流式响应格式

每个 chunk 的格式：

```json
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "delta": {
                "content": "部"
            },
            "finish_reason": null
        }
    ]
}
```

最后一个 chunk：

```json
{
    "choices": [
        {
            "delta": {},
            "finish_reason": "stop"
        }
    ]
}
```

## 3.5 API 认证

为 API 服务添加认证机制。

### 启动时设置 API Key

```bash
# 命令行参数
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --api-key token-abc123

# 或使用环境变量
export VLLM_API_KEY=token-abc123
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

### 客户端使用 API Key

```python
from openai import OpenAI

client = OpenAI(
    api_key="token-abc123",              # 使用设置的 API key
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### curl 使用 API Key

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token-abc123" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### 多 Key 支持

```bash
# 允许多个 key
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --api-key token-abc123,token-xyz789
```

## 3.6 vLLM CLI 工具

vLLM 提供命令行工具，方便日常使用。

### 查看可用模型

```bash
vllm list
```

### 离线补全

```bash
# 基础用法
vllm complete \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompt "The capital of France is"

# 带参数
vllm complete \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompt "The capital of France is" \
    --max-tokens 50 \
    --temperature 0.7
```

### 聊天

```bash
vllm chat \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --message "Hello, how are you?"
```

### 交互模式

```bash
# 启动交互式聊天
vllm chat --model Qwen/Qwen2.5-1.5B-Instruct

# 支持多轮对话
# >>> 你好
# 助手: 你好！有什么可以帮助你的吗？
# >>> 解释一下量子计算
# 助手: 量子计算是一种...
```

## 3.7 请求参数

vLLM 支持 OpenAI 标准参数和 vLLM 扩展参数。

### OpenAI 标准参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | string | 模型名称 |
| `messages` | array | 消息列表（Chat） |
| `prompt` | string/array | 提示文本（Completion） |
| `temperature` | float | 温度参数 |
| `top_p` | float | 核采样阈值 |
| `max_tokens` | integer | 最大 token 数 |
| `stream` | boolean | 流式输出 |
| `stop` | string/array | 停止词 |
| `seed` | integer | 随机种子 |
| `n` | integer | 生成数量 |
| `logprobs` | integer | 返回 logprobs 数量 |

### vLLM 扩展参数

通过 `extra_body` 传递：

```python
response = client.completions.create(
    model="Qwen/Qween2.5-1.5B-Instruct",
    prompt="The capital of France is",
    max_tokens=50,
    extra_body={
        # vLLM 扩展参数
        "repetition_penalty": 1.2,           # 重复惩罚
        "top_k": 50,                         # Top-K 采样
        "min_p": 0.1,                        # 最小概率
        "include_stop_str_in_metric": False,
        "guided_options_cjk_frequency": None,
    }
)
```

### 完整参数示例

```python
# 复杂请求示例
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业的Python编程导师。"},
        {"role": "user", "content": "解释一下装饰器是什么？"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=500,
    n=1,
    stream=False,
    stop=None,
    seed=42,
    extra_body={
        "repetition_penalty": 1.1,
        "top_k": 50,
    }
)

print(response.choices[0].message.content)
```

## 3.8 服务管理

### 使用 systemd 管理服务

创建服务文件 `/etc/systemd/system/vllm.service`：

```ini
[Unit]
Description=vLLM API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/vllm
ExecStart=/opt/vllm/venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-secret-key
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

管理服务：

```bash
# 重新加载配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start vllm

# 设置开机启动
sudo systemctl enable vllm

# 查看状态
sudo systemctl status vllm

# 查看日志
sudo journalctl -u vllm -f
```

### 使用 Docker Compose 部署

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-api
    ports:
      - "8000:8000"
    gpus: all
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - VLLM_API_KEY=your-secret-key
    command: vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0
    restart: unless-stopped
    ipc: host
```

启动：

```bash
docker-compose up -d
```

### 健康检查

```bash
# 简单检查
curl http://localhost:8000/health

# 查看指标
curl http://localhost:8000/metrics
```

### 日志管理

```bash
# 实时查看日志
journalctl -u vllm -f

# 查看最近日志
journalctl -u vllm -n 100

# Docker 日志
docker logs -f vllm-api
```

---

恭喜完成在线服务部署学习！vLLM 的 API 服务部署简单高效，建议进一步学习：

- **生产环境优化**：配置负载均衡、监控告警
- **安全加固**：配置 TLS 证书、限流策略
- **多模型部署**：使用 vLLM 的模型路由器

祝你部署顺利！
