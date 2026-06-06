# vLLM 高级指南 9：多模态模型使用

## 9.1 支持的多模态模型

### 模型类型概览

vLLM 支持多种多模态模型，能够处理图像、音频等多种输入类型：

| 类型 | 模型示例 | 输入格式 | 主要用途 |
|------|---------|---------|---------|
| 视觉-语言 | LLaVA, Qwen-VL, Phi-3-Vision | 图片 + 文本 | 图像理解、视觉问答 |
| 图像描述 | BakLLaVA, CogAgent | 图片 | 图像描述生成 |
| 多模态聊天 | Pixtral, CogVLM, InternVL | 图片 + 对话 | 多轮视觉对话 |
| 语音识别 | Whisper, SenseVoice | 音频 | 语音转文字 |
| 语音合成 | -- | 文本 | 文字转语音（规划中） |

### 常用视觉语言模型

| 模型 | 参数量 | 图像输入 | 上下文长度 | 特点 |
|------|--------|---------|-----------|------|
| Qwen2.5-VL-7B-Instruct | 7B | ✓ | 32768 | 阿里开源，支持多种分辨率 |
| LLaVA-1.6-7B | 7B | ✓ | 4096 | 广泛使用，社区活跃 |
| Phi-3-Vision-128K | 4B | ✓ | 128k | 微软出品，小而精 |
| InternVL2-8B | 8B | ✓ | 32768 | 智谱开源，中文优化 |
| Pixtral-12B | 12B | ✓ | 128k | Mistral 出品 |

## 9.2 图像输入基础

### 基本图像输入

使用 OpenAI 兼容 API 传入图像：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# 单张图片
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                {"type": "text", "text": "这张图片里有什么？"}
            ]
        }
    ]
)

print(completion.choices[0].message.content)
```

### API 参数说明

```python
{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",  # 模型名称
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",  # 图像输入类型
                    "image_url": {
                        "url": "https://example.com/image.jpg",  # 图像 URL
                        "detail": "auto"  # 可选: "low", "high", "auto"
                    }
                },
                {
                    "type": "text",  # 文本输入类型
                    "text": "这张图片里有什么？"
                }
            ]
        }
    ],
    "max_tokens": 1024,  # 最大生成 token 数
    "temperature": 0.7   # 采样温度
}
```

### 图像 detail 参数

| 值 | 说明 | 适用场景 |
|----|------|---------|
| "auto" | 模型自动选择 | 默认推荐 |
| "low" | 低分辨率处理 | 快速响应、简单图像 |
| "high" | 高分辨率处理 | 需要细节的任务 |

## 9.3 本地图片处理

### Base64 编码方式

当图像无法通过 URL 访问时，使用 base64 编码：

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# 读取本地图片并转为 base64
with open("image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "描述这张图片"
                }
            ]
        }
    ]
)
```

### 不同图片格式的 Data URI

```python
# JPEG 格式
data_uri_jpeg = f"data:image/jpeg;base64,{base64_data}"

# PNG 格式
data_uri_png = f"data:image/png;base64,{base64_data}"

# GIF 格式
data_uri_gif = f"data:image/gif;base64,{base64_data}"

# WebP 格式
data_uri_webp = f"data:image/webp;base64,{base64_data}"
```

### 辅助函数

```python
def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为 base64 字符串"""
    import base64
    
    # 根据扩展名确定 MIME 类型
    ext = image_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp',
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def build_image_content(image_source: str, text: str) -> list:
    """构建多模态消息内容"""
    if image_source.startswith('http'):
        # URL 方式
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_source}
        }
    else:
        # 本地文件方式
        img_base64 = encode_image_to_base64(image_source)
        ext = image_source.lower().split('.')[-1]
        mime_type = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
        }
    
    return [
        image_content,
        {"type": "text", "text": text}
    ]
```

## 9.4 多张图片处理

### 多图输入示例

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
                {"type": "text", "text": "比较这两张图片的异同"}
            ]
        }
    ]
)

print(completion.choices[0].message.content)
```

### 实际应用场景

**场景 1：图片比较**

```python
# 电商商品对比
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/product_a.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/product_b.jpg"}},
            {"type": "text", "text": "对比这两款产品的外观设计差异"}
        ]
    }]
)
```

**场景 2：文档图片批量理解**

```python
# 理解多页文档
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/page1.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/page2.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/page3.png"}},
            {"type": "text", "text": "总结这三页文档的主要内容"}
        ]
    }]
)
```

## 9.5 音频输入

### 音频模型支持

vLLM 支持语音识别模型，需要安装额外依赖：

```bash
# 安装音频依赖
pip install vllm[audio]

# 或单独安装
pip install openai-whisper
```

### 语音转文字

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# 读取音频文件
with open("audio.mp3", "rb") as f:
    audio_file = f.read()

# 语音识别
transcription = client.audio.transcriptions.create(
    model="openai/whisper-large-v3-turbo",
    file=("audio.mp3", audio_file, "audio/mpeg"),
    language="zh",  # 指定语言可提高准确率
)

print(transcription.text)
```

### Whisper 模型使用

| 模型 | 参数量 | 速度 | 准确性 |
|------|--------|------|--------|
| whisper-tiny | 39M | 最快 | 较低 |
| whisper-base | 74M | 快 | 中等 |
| whisper-small | 244M | 中等 | 较高 |
| whisper-medium | 769M | 较慢 | 高 |
| whisper-large-v3 | 1550M | 慢 | 最高 |
| whisper-large-v3-turbo | 809M | 中等 | 高 |

### 批量音频处理

```python
import os
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

audio_dir = "./audio_files"
results = []

for filename in os.listdir(audio_dir):
    if filename.endswith(('.mp3', '.wav', '.m4a')):
        filepath = os.path.join(audio_dir, filename)
        
        with open(filepath, "rb") as f:
            audio_content = f.read()
        
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=(filename, audio_content, f"audio/{filename.split('.')[-1]}"),
        )
        
        results.append({
            "file": filename,
            "text": transcription.text
        })
        print(f"{filename}: {transcription.text[:50]}...")
```

## 9.6 离线多模态推理

### 使用 LLM 类直接调用

```python
from vllm import LLM, LLMInputs, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=8192,
)

# 准备多模态输入
inputs = LLMInputs(
    prompt=[
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "这张图片描述了什么场景？"}
    ],
    multi_modal_data={
        "image": "https://example.com/image.jpg"
    }
)

outputs = llm.generate(inputs, SamplingParams(max_tokens=200))
print(outputs[0].outputs[0].text)
```

### 批量推理

```python
from vllm import LLM, LLMInputs, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=8192,
    tensor_parallel_size=2,
)

# 准备多个多模态输入
inputs_list = [
    LLMInputs(
        prompt=[
            {"type": "image_url", "image_url": {"url": f"https://example.com/image{i}.jpg"}},
            {"type": "text", "text": f"描述这张图片 (图片{i})"}
        ],
        multi_modal_data={
            "image": f"https://example.com/image{i}.jpg"
        }
    )
    for i in range(4)
]

sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
outputs = llm.generate(inputs_list, sampling_params)

for i, output in enumerate(outputs):
    print(f"Image {i}: {output.outputs[0].text}")
```

## 9.7 多模态配置参数

### 关键配置项

```python
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    
    # 模型上下文长度
    max_model_len=8192,
    
    # 多模态特定配置
    image_input_type="pixel_values",  # 像素值输入
    image_token_id=None,  # 图像 token ID
    
    # 图像处理配置
    image_processing_batch_size=1,  # 批量处理图片数
    
    # 其他优化
    gpu_memory_utilization=0.85,
)
```

### 配置参数详解

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_input_type` | 图像输入类型 | model 决定 |
| `image_token_id` | 图像 token ID | None |
| `image_processing_batch_size` | 图像处理批次大小 | 1 |
| `image_feature_size` | 图像特征维度 | model 决定 |

### 多模态模型的量化

```python
# INT8 量化
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization="awq",  # AWQ 量化
    dtype="half",
)

# GPTQ 量化
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization="gptq",
    dtype="half",
)
```

## 9.8 多模态前缀缓存

### 多模态缓存机制

多模态输入的前缀缓存与纯文本略有不同：

```python
# 图片被转换为 token 占位符，通过 image_hash 区分
messages = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
        {"type": "text", "text": "这张图片里有什么？"}
    ]}
]

# 缓存键 = hash(文本前缀) + hash(图片内容)
# 不同图片会产生不同的 image_hash，实现精确缓存
```

### 缓存复用示例

```python
# 第一次请求 - 计算并缓存
response1 = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "这张图片有什么物体？"}
        ]}
    ]
)

# 第二次请求 - 相同图片，不同问题 → 复用图片缓存
response2 = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "这张图片是什么场景？"}
        ]}
    ]
)
```

## 9.9 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 图片加载失败 | URL 不可访问或格式错误 | 检查 URL 可访问性，或使用 base64 本地图片 |
| OOM (显存不足) | 图片太大或模型太长 | 减小 `max_model_len`，或使用更小的图片尺寸/分辨率 |
| LoRA 不支持 | 多模态模型的 LoRA 限制 | LoRA 仅适用于语言骨干网络 |
| 图片方向错误 | EXIF 信息问题 | 预处理图片，或使用 PIL 旋转 |
| 批量处理慢 | 串行处理图片 | 增加 `image_processing_batch_size` |
| 输出截断 | max_tokens 太小 | 增加 `max_tokens` 值 |

### 图片预处理建议

```python
from PIL import Image
import requests
from io import BytesIO

def preprocess_image(url: str, max_size: int = 2048) -> Image.Image:
    """预处理图片：调整大小、去除 EXIF"""
    if url.startswith('http'):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(url)
    
    # 去除 EXIF 方向信息
    img = img.convert('RGB')
    
    # 调整大小
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    return img
```

### OOM 优化策略

```python
# 策略 1: 降低图像分辨率
messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "large_image.jpg", "detail": "low"}},
        {"type": "text", "text": "图片里有什么？"}
    ]
}]

# 策略 2: 减小 max_model_len
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=4096,  # 从 8192 降低
)

# 策略 3: 降低 GPU 显存使用
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    gpu_memory_utilization=0.7,  # 从 0.9 降低
)
```

## 9.10 实际应用示例

### 应用 1：OCR + 理解

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# 发票 OCR + 信息提取
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/invoice.jpg"}},
            {"type": "text", "text": """请从这张发票图片中提取：
1. 发票号码
2. 开票日期
3. 购买方名称
4. 销售方名称
5. 商品明细和金额
6. 总金额
以 JSON 格式输出。"""}
        ]
    }]
)

print(completion.choices[0].message.content)
```

### 应用 2：图表分析

```python
# 图表理解
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
            {"type": "text", "text": "分析这个图表，描述数据趋势、主要发现和结论"}
        ]
    }]
)
```

### 应用 3：视觉问答

```python
# 通用视觉问答
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/scene.jpg"}},
            {"type": "text", "text": "请详细描述这张图片的内容，包括场景、人物、物体等"}
        ]
    }]
)
```

### 应用 4：代码生成（UI 设计转代码）

```python
# UI 设计图转代码
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/ui_design.png"}},
            {"type": "text", "text": """根据这个 UI 设计图，生成对应的 HTML 和 CSS 代码。
要求：
1. 使用 Tailwind CSS
2. 实现响应式布局
3. 保持设计比例和风格"""}
        ]
    }]
)
```

## 9.11 多模态模型部署

### 单 GPU 部署

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### 多 GPU 部署（张量并行）

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

### 量化部署

```bash
# AWQ 量化
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-GPTQ \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000
```

### 部署检查

```bash
# 检查模型是否正确加载
curl http://localhost:8000/v1/models

# 测试图像输入
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/test.jpg"}},
        {"type": "text", "text": "测试"}
      ]
    }]
  }'
```
