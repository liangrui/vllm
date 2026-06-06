# vLLM 指南 4：聊天接口与采样参数调优

本指南介绍如何使用 vLLM 构建聊天应用以及优化采样参数。

## 4.1 Chat Template 机制

Chat template 是将对话消息转换为模型输入格式的模板。不同模型使用不同的 template 格式，vLLM 会自动加载模型的默认 template。

### 查看模型 Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print(tokenizer.chat_template)
```

输出示例（Qwen2.5）：

```
{% for message in messages %}
{{ '<|' + message['role_name'] + '|>\n' + message['content'] + '<|end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

### 应用 Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
]

# 生成带 generation prompt 的文本
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 不 tokenize，直接返回字符串
    add_generation_prompt=True  # 添加助手生成提示
)
print(text)
```

输出：

```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is AI?<|end|>
<|assistant|>
```

## 4.2 自定义 Chat Template

### 使用场景

当需要：
- 支持自定义角色
- 调整输出格式
- 适配特定应用场景

### 启动时指定自定义 Template

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --chat-template ./my_template.jinja
```

### 自定义模板示例

```jinja
{# my_template.jinja #}
{% for message in messages %}
{{ '<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

### 常用模板格式

#### Llama 3 格式

```jinja
{% for message in messages %}
{% if loop.first and messages[0]['role'] == 'system' %}
{{ message['content'] }}
{% else %}
{{ '<|begin_of_text|><|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}
```

#### ChatML 格式

```jinja
{% for message in messages %}
{{ '<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

## 4.3 采样参数场景指南

采样参数直接影响生成文本的多样性和确定性。

### 核心参数说明

| 参数 | 作用 | 取值范围 | 影响 |
|------|------|---------|------|
| temperature | 温度参数 | 0.0-2.0 | 越高越随机，越低越确定 |
| top_p | 核采样阈值 | 0.0-1.0 | 越高保留的词越多，越多样化 |
| top_k | Top-K 采样 | 1-1000 | 限制候选词数量 |
| repetition_penalty | 重复惩罚 | 1.0-2.0 | 越高越抑制重复 |

### 场景推荐配置

| 场景 | temperature | top_p | top_k | repetition_penalty |
|------|-------------|-------|-------|-------------------|
| 精确问答 | 0.0-0.3 | 0.9-1.0 | 20-50 | 1.0-1.1 |
| 代码生成 | 0.0-0.3 | 0.95 | 50 | 1.0-1.05 |
| 创意写作 | 0.7-1.0 | 0.9 | 100 | 1.0-1.2 |
| 摘要生成 | 0.3-0.6 | 0.9 | 50 | 1.0 |
| 翻译 | 0.3-0.7 | 0.9 | 50 | 1.0 |
| 头脑风暴 | 0.8-1.2 | 0.95 | 100 | 1.0-1.1 |

### 参数组合策略

#### 高确定性场景（问答、代码）

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.0,  # 完全确定性
    top_p=1.0,
    top_k=-1,  # 禁用 top_k 限制
    repetition_penalty=1.0,
)
```

#### 平衡模式（日常对话）

```python
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
)
```

#### 高随机性场景（创意写作）

```python
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.85,
    top_k=100,
    repetition_penalty=1.15,
)
```

## 4.4 对话历史管理

### 基础对话实现

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    outputs = llm.chat(messages, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text
    
    messages.append({"role": "assistant", "content": response})
    print(f"Assistant: {response}")
```

### 带 Token 计数的管理

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def count_tokens(messages):
    """计算消息列表的 token 数量"""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text))

# 添加用户消息
messages.append({"role": "user", "content": user_input})
current_tokens = count_tokens(messages)
print(f"Current tokens: {current_tokens}")
```

## 4.5 多轮对话优化

### 对话历史截断

当对话过长时，需要截断早期消息以适应模型的最大上下文长度：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

MAX_TOKENS = 4096

def count_tokens(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text))

def truncate_history(messages, max_tokens=MAX_TOKENS):
    """截断过长的对话历史，保留 system prompt"""
    while count_tokens(messages) > max_tokens:
        # 确保至少保留 system prompt 和一对对话
        if len(messages) <= 3:
            break
        # 移除最旧的用户-助手对（index 1 和 2）
        messages.pop(1)  # 移除第一个用户消息
        messages.pop(1)  # 移除第一个助手回复
    return messages

# 使用示例
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]
```

### 智能截断策略

```python
def smart_truncate(messages, max_tokens=MAX_TOKENS, preserve_turns=2):
    """智能截断，保留最近的对话"""
    if count_tokens(messages) <= max_tokens:
        return messages
    
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    
    # 保留最近的对话
    recent_messages = messages[-(preserve_turns * 2):]
    
    new_messages = []
    if system_msg:
        new_messages.append(system_msg)
    new_messages.extend(recent_messages)
    
    # 如果还是太长，渐进式截断
    while count_tokens(new_messages) > max_tokens and len(new_messages) > 3:
        new_messages.pop(1)  # 移除最旧的对话
    
    return new_messages
```

### 不同截断策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 简单移除最早对话 | 实现简单 | 可能丢失重要上下文 | 短对话 |
| 保留最近对话 | 上下文相关性强 | 可能丢失早期关键信息 | 长对话 |
| 摘要+保留 | 保留全局信息 | 需要额外 LLM 调用 | 超长对话 |

## 4.6 系统提示词工程

### 角色设定最佳实践

```python
# 技术助手
tech_assistant_prompt = """你是一位资深的Python工程师。
- 擅长编写清晰、高效的代码
- 遵循PEP 8规范
- 解释技术概念时使用简单易懂的语言
- 代码示例要包含注释"""

# 翻译助手
translator_prompt = """你是一位专业翻译，精通中英文互译。
- 翻译准确流畅，保持原文风格
- 注意中英文表达习惯差异
- 适当本地化，而非逐字翻译"""

# 代码审查助手
code_reviewer_prompt = """你是一位资深代码审查员。
- 关注代码质量、可读性和性能
- 提出具体改进建议，带代码示例
- 提供业界最佳实践参考
- 评估测试覆盖情况"""

# 创意写作助手
writer_prompt = """你是一位创意写作专家。
- 擅长多种文体：小说、散文、诗歌等
- 文笔优美，情感细腻
- 善于设置悬念和情节转折"""

# 角色使用示例
messages = [
    {"role": "system", "content": tech_assistant_prompt},
    {"role": "user", "content": "如何优化 Python 代码的性能？"},
]
```

### Few-shot Prompting

```python
# 提供示例来引导模型输出
few_shot_messages = [
    {"role": "system", "content": "你是一个格式化助手，将用户输入转换为 JSON 格式。"},
    {"role": "user", "content": "苹果是红色的，甜甜的"},
    {"role": "assistant", "content": '{"name": "苹果", "color": "红色", "taste": "甜"}'},
    {"role": "user", "content": "香蕉是黄色的，软糯的"},
    {"role": "assistant", "content": '{"name": "香蕉", "color": "黄色", "taste": "软糯"}'},
    {"role": "user", "content": "柠檬是黄色的，酸酸的"},
]
```

### 提示词结构模板

```
[角色定义] + [能力说明] + [输出格式要求] + [约束条件]

示例：
你是一位数据分析师（角色）。你的职责是分析销售数据并生成报告（能力）。
请用表格形式呈现数据，用中文撰写分析报告（格式）。不要臆测数据，只基于提供的信息分析（约束）。
```

## 4.7 结构化输出

### 使用 guided_json 强制 JSON 输出

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 定义 JSON schema
completion = llm.chat(
    messages=[{"role": "user", "content": "提取水果信息：苹果是红色的，甜甜的；香蕉是黄色的。"}],
    sampling_params=SamplingParams(temperature=0.0),
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "fruits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "水果名称"},
                            "color": {"type": "string", "description": "颜色"},
                            "taste": {"type": "string", "description": "味道"}
                        },
                        "required": ["name", "color", "taste"]
                    }
                }
            },
            "required": ["fruits"]
        }
    }
)

print(completion[0].outputs[0].text)
```

### 常用 Output Schema

#### 简单对象

```python
"guided_json": {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    }
}
```

#### 带嵌套的对象

```python
"guided_json": {
    "type": "object",
    "properties": {
        "users": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "roles": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "email"]
            }
        }
    }
}
```

#### 枚举类型

```python
"guided_json": {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "score": {"type": "number", "minimum": 0, "maximum": 1}
    }
}
```

### 结合 OpenAI API 格式

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1"
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "列出三种编程语言"}],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["languages"]
        }
    }
)

import json
result = json.loads(completion.choices[0].message.content)
print(result)
```

### 输出格式验证

```python
import json
from pydantic import BaseModel, ValidationError
from typing import List

class Fruit(BaseModel):
    name: str
    color: str
    taste: str

class FruitList(BaseModel):
    fruits: List[Fruit]

def validate_output(text: str) -> FruitList:
    """验证并解析 JSON 输出"""
    try:
        data = json.loads(text)
        return FruitList(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")

# 使用
result = validate_output(completion[0].outputs[0].text)
print(result.fruits[0].name)
```
