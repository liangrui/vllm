# 医学术语偏好 Logits Processor

## 概述

这是一个为 vLLM 设计的自定义 logits processor，用于在语音转文本（ASR）等任务中，使模型生成的文本更倾向于包含医学相关的词汇。

## 功能特性

1. **中英文支持**: 内置中文和英文的医学术语库
2. **自定义术语**: 支持自定义医学术语列表
3. **可调偏置**: 可调整医学术语的 logits 增加值
4. **两种实现**: 
   - v1 API 批量处理版本 (MedicalTermLogitsProcessor)
   - 请求级版本 (MedicalTermRequestLevelLogitsProcessor)

## 文件结构

```
vllm/
└── plugins/
    └── logits_processors/
        ├── __init__.py
        ├── medical_terms_processor.py  # 核心实现
        └── medical_terms_list.py       # 扩展术语列表

examples/
└── medical_terms_example.py          # 使用示例

tests/
└── test_medical_terms_processor.py   # 测试脚本
```

## 快速开始

### 1. 请求级别使用（推荐用于简单场景）

```python
from vllm import LLM, SamplingParams
from vllm.plugins.logits_processors import (
    MedicalTermRequestLevelLogitsProcessor,
    EXTENDED_CHINESE_MEDICAL_TERMS
)
from transformers import AutoTokenizer

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-small")

# 创建医学术语 processor
medical_processor = MedicalTermRequestLevelLogitsProcessor(
    tokenizer=tokenizer,
    bias=5.0,  # 医学术语的 logits 增加值
    language="zh",  # 中文
    medical_terms=EXTENDED_CHINESE_MEDICAL_TERMS  # 使用扩展术语列表
)

# 创建 SamplingParams
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    logits_processors=[medical_processor]
)

# 进行推理
llm = LLM(model="openai/whisper-small")
outputs = llm.generate(prompts=your_audio_prompts, sampling_params=sampling_params)
```

### 2. v1 API 使用

```python
from vllm import LLM, SamplingParams

# 创建 SamplingParams，启用医学术语偏好
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    extra_args={
        'enable_medical_bias': True,
        'medical_bias': 5.0,
        'language': 'zh'
    }
)

# 初始化 vLLM 时加载自定义 processor
llm = LLM(
    model="openai/whisper-small",
    logits_processors=["vllm.plugins.logits_processors.MedicalTermLogitsProcessor"]
)

# 进行推理
outputs = llm.generate(prompts=your_audio_prompts, sampling_params=sampling_params)
```

## 配置参数

### SamplingParams.extra_args 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enable_medical_bias | bool | False | 是否启用医学术语偏好 |
| medical_bias | float | 5.0 | 医学术语的 logits 增加值 |
| medical_terms | list[str] | None | 自定义医学术语列表 |
| language | str | 'zh' | 语言设置，'zh' 或 'en' |

## 内置术语库

### 基础术语

- **中文**: 50+ 常用医学术语
- **英文**: 50+ 常用医学术语

### 扩展术语

```python
from vllm.plugins.logits_processors import (
    EXTENDED_CHINESE_MEDICAL_TERMS,  # 120+ 中文术语
    EXTENDED_ENGLISH_MEDICAL_TERMS   # 130+ 英文术语
)
```

## 自定义术语示例

```python
# 心血管专科术语
cardiology_terms = [
    "心肌梗塞", "冠心病", "心绞痛", "心律失常", "房颤",
    "心力衰竭", "心脏骤停", "支架", "搭桥手术",
    "心电图", "超声心动图", "冠脉造影",
    "阿司匹林", "氯吡格雷", "他汀", "降压药",
    "胸痛", "胸闷", "心悸", "气短", "水肿"
]

# 使用自定义术语
processor = MedicalTermRequestLevelLogitsProcessor(
    tokenizer=tokenizer,
    medical_terms=cardiology_terms,
    bias=8.0
)
```

## 运行示例

```bash
# 运行使用示例
python examples/medical_terms_example.py

# 运行测试
python tests/test_medical_terms_processor.py
```

## 适用场景

1. **医疗咨询转录**: 将医患对话转换为文字时，提高医学术语识别率
2. **病历记录**: 自动转录医生口述的病历
3. **医学教育**: 医学课程视频/音频的字幕生成
4. **远程医疗**: 在线问诊的实时语音转文字

## 注意事项

1. **设置合理的 bias 值**: 
   - 建议范围: 3.0 - 10.0
   - 值太大可能导致过度偏向医学术语
2. **选择合适的术语**: 根据具体应用场景选择或自定义术语列表
3. **Tokenizer 匹配**: 确保使用与模型匹配的 tokenizer

## 参考

- vLLM 自定义 Logits Processors 文档: https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
