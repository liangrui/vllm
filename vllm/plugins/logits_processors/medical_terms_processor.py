# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizer

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MedicalTermLogitsProcessor(LogitsProcessor):
    """医学术语偏好的logits processor
    
    该processor用于在语音转文本（ASR）任务中，通过提升医学相关词汇的logits值，
    使得生成的文本更倾向于包含医学术语。
    """
    
    # 预定义的中文医学术语列表
    DEFAULT_MEDICAL_TERMS = [
        "体温", "脉搏", "血压", "心率", "呼吸", "血糖", "胆固醇",
        "心电图", "X光", "CT", "MRI", "B超",
        "手术", "治疗", "诊断", "处方", "药物", "剂量",
        "内科", "外科", "儿科", "妇科", "骨科", "神经科",
        "医生", "护士", "患者", "病历", "住院", "门诊",
        "感染", "炎症", "肿瘤", "癌症", "糖尿病", "高血压",
        "症状", "疼痛", "发热", "咳嗽", "头痛", "恶心",
        "疫苗", "抗体", "免疫", "遗传", "基因",
        "康复", "护理", "保健", "体检", "复查"
    ]
    
    # 预定义的英文医学术语列表
    DEFAULT_ENGLISH_MEDICAL_TERMS = [
        "temperature", "pulse", "blood pressure", "heart rate", "respiration",
        "blood sugar", "cholesterol",
        "ECG", "EKG", "X-ray", "CT", "MRI", "ultrasound",
        "surgery", "treatment", "diagnosis", "prescription", "medication", "dosage",
        "internal medicine", "surgery", "pediatrics", "gynecology", 
        "orthopedics", "neurology",
        "doctor", "physician", "nurse", "patient", "medical record",
        "hospitalization", "outpatient",
        "infection", "inflammation", "tumor", "cancer", 
        "diabetes", "hypertension",
        "symptom", "pain", "fever", "cough", "headache", "nausea",
        "vaccine", "antibody", "immune", "genetic", "gene",
        "rehabilitation", "nursing", "health care", "check-up", "follow-up"
    ]

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.tokenizer = None
        self.vocab_size = None
        
        # 存储每个请求的配置
        # index -> (bias_value, medical_token_ids, active)
        self.req_configs: dict[int, tuple[float, list[int], bool]] = {}
        
        # 用于apply的tensor存储
        self.req_indices: torch.Tensor = torch.tensor([], dtype=torch.int64, device=device)
        self.token_indices: torch.Tensor = torch.tensor([], dtype=torch.int64, device=device)
        self.bias_values: torch.Tensor = torch.tensor([], dtype=torch.float32, device=device)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """验证参数
        
        支持的extra_args参数:
        - enable_medical_bias: bool - 是否启用医学术语偏好
        - medical_bias: float - 医学术语的logits增加值，默认为5.0
        - medical_terms: list[str] - 自定义医学术语列表
        - language: str - 语言设置，"zh" 或 "en"，默认"zh"
        """
        extra_args: dict[str, Any] = sampling_params.extra_args or {}
        
        # 验证medical_bias
        if "medical_bias" in extra_args:
            bias = extra_args["medical_bias"]
            if not isinstance(bias, (int, float)):
                raise ValueError(f"medical_bias必须是数值类型，得到: {type(bias)}")
        
        # 验证medical_terms
        if "medical_terms" in extra_args:
            terms = extra_args["medical_terms"]
            if not isinstance(terms, list):
                raise ValueError(f"medical_terms必须是列表类型，得到: {type(terms)}")
            for term in terms:
                if not isinstance(term, str):
                    raise ValueError(f"medical_terms中的每个元素必须是字符串，得到: {type(term)}")
        
        # 验证language
        if "language" in extra_args:
            lang = extra_args["language"]
            if lang not in ["zh", "en"]:
                raise ValueError(f"language必须是'zh'或'en'，得到: {lang}")

    def is_argmax_invariant(self) -> bool:
        """此processor会改变argmax结果"""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return
        
        needs_update = False
        
        # 处理新增的请求
        for index, params, _, _ in batch_update.added:
            extra_args: dict[str, Any] = params.extra_args or {}
            
            # 检查是否启用医学术语偏好
            enable = extra_args.get("enable_medical_bias", False)
            
            if enable:
                # 初始化tokenizer（如果还没有）
                if self.tokenizer is None:
                    self._init_tokenizer(params)
                
                bias = extra_args.get("medical_bias", 5.0)
                custom_terms = extra_args.get("medical_terms")
                language = extra_args.get("language", "zh")
                
                # 获取医学术语的token ids
                medical_token_ids = self._get_medical_token_ids(
                    custom_terms=custom_terms,
                    language=language
                )
                
                self.req_configs[index] = (bias, medical_token_ids, True)
                needs_update = True
            else:
                if index in self.req_configs:
                    del self.req_configs[index]
                    needs_update = True
        
        if self.req_configs:
            # 处理删除的请求
            for index in batch_update.removed:
                if index in self.req_configs:
                    del self.req_configs[index]
                    needs_update = True
            
            # 处理移动的请求
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_configs.pop(adx, None)
                
                if direct == MoveDirectionality.SWAP:
                    b_val = self.req_configs.pop(bdx, None)
                    if b_val is not None:
                        self.req_configs[adx] = b_val
                        needs_update = True
                
                if a_val is not None:
                    self.req_configs[bdx] = a_val
                    needs_update = True
        
        # 更新tensor
        if needs_update:
            self._update_tensors()

    def _init_tokenizer(self, params: SamplingParams):
        """初始化tokenizer（延迟初始化）"""
        # 这里我们假设tokenizer可以从全局获取
        # 在实际使用中，可能需要通过vllm_config获取
        # 暂时使用占位符，实际使用时需要正确初始化
        pass

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """外部设置tokenizer的方法"""
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

    def _get_medical_token_ids(self, custom_terms: list[str] | None, language: str) -> list[int]:
        """获取医学术语的token ids"""
        if self.tokenizer is None:
            return []
        
        # 选择默认术语
        if custom_terms is not None:
            terms = custom_terms
        elif language == "en":
            terms = self.DEFAULT_ENGLISH_MEDICAL_TERMS
        else:
            terms = self.DEFAULT_MEDICAL_TERMS
        
        # 将术语转换为token ids
        token_ids = set()
        for term in terms:
            # 对每个术语进行编码
            ids = self.tokenizer.encode(term, add_special_tokens=False)
            token_ids.update(ids)
        
        return list(token_ids)

    def _update_tensors(self):
        """更新用于apply的tensors"""
        req_indices: list[int] = []
        token_indices: list[int] = []
        bias_values: list[float] = []
        
        for req_idx, (bias, tok_ids, _) in self.req_configs.items():
            for tok_id in tok_ids:
                req_indices.append(req_idx)
                token_indices.append(tok_id)
                bias_values.append(bias)
        
        self.req_indices = torch.tensor(
            req_indices, dtype=torch.int64, device=self.device
        )
        self.token_indices = torch.tensor(
            token_indices, dtype=torch.int64, device=self.device
        )
        self.bias_values = torch.tensor(
            bias_values, dtype=torch.float32, device=self.device
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用logits bias"""
        if self.req_indices.numel() > 0:
            logits[self.req_indices, self.token_indices] += self.bias_values
        return logits


class MedicalTermRequestLevelLogitsProcessor:
    """请求级别的医学术语logits processor
    
    这是一个简单的、非v1版本的实现，适用于需要独立处理每个请求的场景。
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        bias: float = 5.0,
        medical_terms: list[str] | None = None,
        language: str = "zh"
    ):
        self.tokenizer = tokenizer
        self.bias = bias
        
        # 选择默认术语
        if medical_terms is not None:
            self.terms = medical_terms
        elif language == "en":
            self.terms = MedicalTermLogitsProcessor.DEFAULT_ENGLISH_MEDICAL_TERMS
        else:
            self.terms = MedicalTermLogitsProcessor.DEFAULT_MEDICAL_TERMS
        
        # 预计算医学术语的token ids
        self.medical_token_ids = set()
        for term in self.terms:
            ids = tokenizer.encode(term, add_special_tokens=False)
            self.medical_token_ids.update(ids)
    
    def __call__(
        self,
        output_token_ids: list[int],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """处理单个请求的logits"""
        # 为医学术语增加bias
        for token_id in self.medical_token_ids:
            if token_id < logits.shape[0]:
                logits[token_id] += self.bias
        return logits
