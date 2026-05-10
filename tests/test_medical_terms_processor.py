# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
医学术语 Logits Processor 测试
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock

import torch
from transformers import AutoTokenizer

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.plugins.logits_processors import (
    MedicalTermLogitsProcessor,
    MedicalTermRequestLevelLogitsProcessor,
    EXTENDED_CHINESE_MEDICAL_TERMS,
    EXTENDED_ENGLISH_MEDICAL_TERMS
)


class TestMedicalTermRequestLevelLogitsProcessor(unittest.TestCase):
    """测试请求级别的医学术语 logits processor"""
    
    @classmethod
    def setUpClass(cls):
        """使用一个简单的 tokenizer 进行测试"""
        # 使用一个小型测试模型的 tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
    
    def test_initialization(self):
        """测试初始化"""
        processor = MedicalTermRequestLevelLogitsProcessor(
            tokenizer=self.tokenizer,
            bias=5.0,
            language="zh"
        )
        
        self.assertIsNotNone(processor.tokenizer)
        self.assertEqual(processor.bias, 5.0)
        self.assertGreater(len(processor.medical_token_ids), 0)
    
    def test_initialization_with_custom_terms(self):
        """测试使用自定义术语初始化"""
        custom_terms = ["血压", "心率", "糖尿病"]
        processor = MedicalTermRequestLevelLogitsProcessor(
            tokenizer=self.tokenizer,
            bias=8.0,
            medical_terms=custom_terms
        )
        
        self.assertEqual(processor.bias, 8.0)
        self.assertEqual(processor.terms, custom_terms)
        self.assertGreater(len(processor.medical_token_ids), 0)
    
    def test_initialization_with_english(self):
        """测试英文术语初始化"""
        processor = MedicalTermRequestLevelLogitsProcessor(
            tokenizer=self.tokenizer,
            language="en"
        )
        
        self.assertEqual(processor.terms, MedicalTermLogitsProcessor.DEFAULT_ENGLISH_MEDICAL_TERMS)
    
    def test_call_method(self):
        """测试 __call__ 方法是否正确增加 logits"""
        processor = MedicalTermRequestLevelLogitsProcessor(
            tokenizer=self.tokenizer,
            bias=5.0,
            medical_terms=["血压"]
        )
        
        # 创建模拟的 logits
        vocab_size = len(self.tokenizer)
        logits = torch.zeros(vocab_size)
        
        # 记录医学术语 token 的原始 logits
        medical_token_ids = processor.medical_token_ids
        original_logits = logits.clone()
        
        # 调用 processor
        output_logits = processor([], logits)
        
        # 验证医学术语的 logits 被增加了
        for token_id in medical_token_ids:
            if token_id < vocab_size:
                self.assertEqual(output_logits[token_id], original_logits[token_id] + 5.0)
    
    def test_extended_terms_list(self):
        """测试扩展术语列表"""
        self.assertGreater(len(EXTENDED_CHINESE_MEDICAL_TERMS), 
                          len(MedicalTermLogitsProcessor.DEFAULT_MEDICAL_TERMS))
        self.assertGreater(len(EXTENDED_ENGLISH_MEDICAL_TERMS), 
                          len(MedicalTermLogitsProcessor.DEFAULT_ENGLISH_MEDICAL_TERMS))


class TestMedicalTermLogitsProcessor(unittest.TestCase):
    """测试 v1 API 版本的医学术语 logits processor"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_vllm_config = Mock()
        self.mock_vllm_config.scheduler_config = Mock()
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        
        self.device = torch.device("cpu")
    
    def test_initialization(self):
        """测试初始化"""
        processor = MedicalTermLogitsProcessor(
            vllm_config=self.mock_vllm_config,
            device=self.device,
            is_pin_memory=False
        )
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.device, self.device)
        self.assertEqual(len(processor.req_configs), 0)
    
    def test_set_tokenizer(self):
        """测试设置 tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
        
        processor = MedicalTermLogitsProcessor(
            vllm_config=self.mock_vllm_config,
            device=self.device,
            is_pin_memory=False
        )
        
        processor.set_tokenizer(tokenizer)
        self.assertEqual(processor.tokenizer, tokenizer)
        self.assertEqual(processor.vocab_size, len(tokenizer))
    
    def test_validate_params(self):
        """测试参数验证"""
        # 创建 mock SamplingParams
        mock_params = Mock()
        
        # 测试有效参数
        mock_params.extra_args = {
            'enable_medical_bias': True,
            'medical_bias': 5.0,
            'medical_terms': ['血压', '心率'],
            'language': 'zh'
        }
        
        # 应该不抛出异常
        MedicalTermLogitsProcessor.validate_params(mock_params)
        
        # 测试无效的 bias 类型
        mock_params.extra_args = {'medical_bias': 'not a number'}
        with self.assertRaises(ValueError):
            MedicalTermLogitsProcessor.validate_params(mock_params)
        
        # 测试无效的 language
        mock_params.extra_args = {'language': 'invalid'}
        with self.assertRaises(ValueError):
            MedicalTermLogitsProcessor.validate_params(mock_params)
    
    def test_is_argmax_invariant(self):
        """测试 argmax invariant 方法"""
        processor = MedicalTermLogitsProcessor(
            vllm_config=self.mock_vllm_config,
            device=self.device,
            is_pin_memory=False
        )
        
        # 这个 processor 会改变 argmax 结果，所以应该返回 False
        self.assertFalse(processor.is_argmax_invariant())


def run_tests():
    """运行测试"""
    print("运行医学术语 Logits Processor 测试...")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalTermRequestLevelLogitsProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalTermLogitsProcessor))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("所有测试通过！")
        return 0
    else:
        print(f"测试失败: {len(result.failures) + len(result.errors)}")
        return 1


if __name__ == "__main__":
    # 检查是否安装了必要的依赖
    try:
        import transformers
    except ImportError:
        print("错误: 需要安装 transformers 库")
        print("运行: pip install transformers")
        sys.exit(1)
    
    sys.exit(run_tests())
