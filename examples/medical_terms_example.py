#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
医学术语偏好 Logits Processor 使用示例

该示例演示如何使用 MedicalTermLogitsProcessor 来让语音转文本（ASR）任务
更倾向于生成医学相关的词汇。
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from transformers import AutoTokenizer
import torch

from vllm import LLM, SamplingParams
from vllm.plugins.logits_processors import (
    MedicalTermRequestLevelLogitsProcessor,
    EXTENDED_CHINESE_MEDICAL_TERMS,
    EXTENDED_ENGLISH_MEDICAL_TERMS
)


def example_request_level():
    """
    示例 1: 使用请求级别的 MedicalTermRequestLevelLogitsProcessor
    
    这是一个简单的实现，适用于离线推理场景。
    """
    print("=" * 60)
    print("示例 1: 使用请求级别的 MedicalTermRequestLevelLogitsProcessor")
    print("=" * 60)
    
    # 加载 tokenizer（以 Whisper 为例）
    model_name = "openai/whisper-small"  # 可以替换为其他支持的模型
    print(f"\n加载 Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建医学术语 logits processor
    print("\n创建医学术语 Logits Processor...")
    medical_processor = MedicalTermRequestLevelLogitsProcessor(
        tokenizer=tokenizer,
        bias=5.0,  # 设置医学术语的 logits 增加值
        language="zh",  # 中文
        # 可以自定义术语列表，或使用预定义的扩展列表
        # medical_terms=EXTENDED_CHINESE_MEDICAL_TERMS
    )
    
    print(f"\n预定义的中文医学术语数量（基础）: {len(MedicalTermRequestLevelLogitsProcessor.DEFAULT_MEDICAL_TERMS)}")
    print(f"扩展的中文医学术语数量: {len(EXTENDED_CHINESE_MEDICAL_TERMS)}")
    print(f"医学术语对应的 token 数量: {len(medical_processor.medical_token_ids)}")
    
    print("\n" + "=" * 60)
    print("提示: 在实际应用中，你可以将此 processor 传递给 vLLM 的 SamplingParams")
    print("使用 logits_processors 参数")
    print("=" * 60)


def example_configuration():
    """
    示例 2: 配置选项说明
    """
    print("\n" + "=" * 60)
    print("示例 2: 配置选项说明")
    print("=" * 60)
    
    print("""
MedicalTermLogitsProcessor 支持以下配置参数（通过 SamplingParams.extra_args）:

1. enable_medical_bias (bool)
   - 是否启用医学术语偏好
   - 默认: False
   - 示例: {'enable_medical_bias': True}

2. medical_bias (float)
   - 医学术语的 logits 增加值
   - 默认: 5.0
   - 更大的值会让医学术语更可能被选中
   - 示例: {'medical_bias': 8.0}

3. medical_terms (list[str])
   - 自定义医学术语列表
   - 默认: 使用内置的中文/英文术语
   - 示例: {'medical_terms': ['心脏病', '高血压', '糖尿病']}

4. language (str)
   - 语言设置，'zh' 或 'en'
   - 默认: 'zh'
   - 示例: {'language': 'en'}
""")


def example_with_v1_api():
    """
    示例 3: v1 API 使用说明
    """
    print("\n" + "=" * 60)
    print("示例 3: v1 API 使用说明")
    print("=" * 60)
    
    print("""
在 vLLM v1 API 中使用 MedicalTermLogitsProcessor 的步骤:

1. 在启动 vLLM 引擎时加载自定义 logits processor:
   - 方法1: 通过 --load-format 或配置文件
   - 方法2: 自动检测安装的插件
   - 方法3: 离线模式下直接传递类对象

2. 创建 SamplingParams 时启用医学术语偏好:
   ```python
   from vllm import SamplingParams
   
   sampling_params = SamplingParams(
       extra_args={
           'enable_medical_bias': True,
           'medical_bias': 5.0,
           'language': 'zh',
           # 可选: 自定义术语列表
           # 'medical_terms': ['体温', '血压', '心率', ...]
       }
   )
   ```

3. 进行推理（例如 ASR 任务）:
   ```python
   from vllm import LLM
   
   llm = LLM(model="openai/whisper-small")
   outputs = llm.generate(
       prompts=your_audio_prompts,
       sampling_params=sampling_params
   )
   ```

注意: 使用 v1 API 时，需要在初始化后调用 set_tokenizer 方法
来设置 tokenizer:
   ```python
   # 在初始化后设置 tokenizer
   processor = MedicalTermLogitsProcessor(...)
   processor.set_tokenizer(tokenizer)
   ```
""")


def example_custom_terms():
    """
    示例 4: 使用自定义医学术语
    """
    print("\n" + "=" * 60)
    print("示例 4: 使用自定义医学术语")
    print("=" * 60)
    
    # 自定义心血管专科术语
    cardiology_terms = [
        "心肌梗塞", "冠心病", "心绞痛", "心律失常", "房颤",
        "心力衰竭", "心脏骤停", "支架", "搭桥手术",
        "心电图", "超声心动图", "冠脉造影",
        "阿司匹林", "氯吡格雷", "他汀", "降压药",
        "胸痛", "胸闷", "心悸", "气短", "水肿"
    ]
    
    print(f"\n自定义心血管专科术语: {len(cardiology_terms)} 个")
    print("术语列表:")
    for term in cardiology_terms:
        print(f"  - {term}")
    
    print("""
使用自定义术语的方法:

1. 请求级别的 processor:
   ```python
   processor = MedicalTermRequestLevelLogitsProcessor(
       tokenizer=tokenizer,
       medical_terms=cardiology_terms,  # 使用自定义列表
       bias=8.0
   )
   ```

2. v1 API:
   ```python
   sampling_params = SamplingParams(
       extra_args={
           'enable_medical_bias': True,
           'medical_terms': cardiology_terms,
           'medical_bias': 8.0
       }
   )
   ```
""")


def example_asr_workflow():
    """
    示例 5: ASR 任务完整工作流程
    """
    print("\n" + "=" * 60)
    print("示例 5: ASR 任务完整工作流程（伪代码）")
    print("=" * 60)
    
    print("""
以下是使用 MedicalTermLogitsProcessor 进行医疗场景 ASR 的完整流程:

```python
import torch
from vllm import LLM, SamplingParams
from vllm.plugins.logits_processors import (
    MedicalTermLogitsProcessor,
    MedicalTermRequestLevelLogitsProcessor,
    EXTENDED_CHINESE_MEDICAL_TERMS
)
from transformers import AutoTokenizer, WhisperProcessor

# 1. 初始化模型和 processor
model_name = "openai/whisper-large-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 准备音频输入
# audio = load_your_audio_file("medical_consultation.wav")
# processor = WhisperProcessor.from_pretrained(model_name)
# input_features = processor(audio, return_tensors="pt").input_features

# 3. 创建 SamplingParams，启用医学术语偏好
# 方法 A: 使用请求级别的 processor（适合简单场景）
medical_processor = MedicalTermRequestLevelLogitsProcessor(
    tokenizer=tokenizer,
    bias=5.0,
    language="zh",
    medical_terms=EXTENDED_CHINESE_MEDICAL_TERMS  # 使用扩展术语
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    logits_processors=[medical_processor]  # 添加到 logits_processors 列表
)

# 方法 B: 使用 v1 API 的自定义 processor
# （需要先配置加载自定义 logits processor）
'''
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    extra_args={
        'enable_medical_bias': True,
        'medical_bias': 5.0,
        'language': 'zh'
    }
)
'''

# 4. 初始化 vLLM 并进行推理
'''
llm = LLM(
    model=model_name,
    # 如果使用 v1 自定义 processor，需要在这里配置加载
    # logits_processors=["vllm.plugins.logits_processors.MedicalTermLogitsProcessor"]
)

outputs = llm.generate(
    prompts=[{"audio": audio_path}],  # 具体格式取决于 vLLM 的 ASR 接口
    sampling_params=sampling_params
)

# 5. 获取并使用结果
for output in outputs:
    transcription = output.outputs[0].text
    print(f"转录结果: {transcription}")
'''

print("ASR 工作流程设置完成！")
```
""")


def main():
    """运行所有示例"""
    print("医学术语偏好 Logits Processor 使用示例")
    print("=" * 60)
    
    example_request_level()
    example_configuration()
    example_with_v1_api()
    example_custom_terms()
    example_asr_workflow()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
