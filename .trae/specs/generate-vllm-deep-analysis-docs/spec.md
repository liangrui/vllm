# vLLM 代码深度分析报告 Spec

## Why
vLLM 是当前最主流的大模型推理框架之一，其代码库庞大且架构复杂（含 v0/v1 双版本演进），缺乏一份系统性的深度技术文档来帮助开发者理解其设计思想、核心实现和扩展机制。需要产出面向系统架构师的高质量分析文档，降低理解和贡献门槛。

## What Changes
- 在 `/workspace/ReadCode/` 目录下创建 **22 个 Markdown 文件**：1 个总览导航文件 + 21 个分模块深度分析文件
- 更新已有的 `vllm_code_structure_analysis.md` 总览文件为 2.0 版本
- 新建 `01_architecture_overview.md` 至 `21_design_patterns.md` 共 21 个分模块文件，覆盖 vLLM 全部核心子系统
- 每个文件包含：Mermaid 架构图、概念解释、核心代码示例（带源码行号引用）、数据流分析、设计决策理由

## Impact
- Affected specs: 无前置 spec 依赖
- Affected code: 仅产出文档，不修改 `/workspace/vllm` 源码；文档输出至 `/workspace/ReadCode/`

## ADDED Requirements

### Requirement: 总览导航文件
系统 SHALL 提供一个总览文件 `vllm_code_structure_analysis.md`，作为整个分析的导航入口和宏观概览。

#### Scenario: 总览文件内容完整性
- **WHEN** 用户打开总览文件
- **THEN** 文件应包含：项目概述、六层 Mermaid 架构图、v0→v1 版本演进路线、目录结构全景图、快速导航索引（链接到全部 21 个分模块文件）、关键技术特性一览表、数据流总览图、设计哲学总结

### Requirement: 分模块分析文件（01-21）
系统 SHALL 为以下 21 个子系统各产出一个独立的深度分析 Markdown 文件：

| 编号 | 模块 | 核心关注点 |
|------|------|-----------|
| 01 | architecture_overview | 分层架构、版本演进、核心数据结构、设计原则 |
| 02 | config_system | VllmConfig 及全部子配置类、环境变量覆盖、OptimizationLevel |
| 03 | engine_core | LLMEngine/AsyncLLM/EngineCore、处理管线、异常机制 |
| 04 | scheduler | Scheduler.schedule() 完整逻辑、Continuous Batching、抢占机制 |
| 05 | pagedattention | Block/BlockTable/BlockPool、内存共享、Prefix Caching |
| 06 | attention_backends | FlashAttention/FlashInfer/MLA/Triton/CPU 等全部后端 |
| 07 | model_executor | ModelRegistry、200+ 模型支持、主要模型家族分析 |
| 08 | worker_and_executor | Executor 抽象、Worker 层、ModelRunner、输入批处理 |
| 09 | kv_cache | KVCacheManager 层次、Offload/Transfer/Prefix Caching |
| 10 | sampling | SamplingParams/Sampler、RejectionSampler、ParallelSampling |
| 11 | multimodal | MultiModalRegistry、媒体连接器、特征提取管线 |
| 12 | quantization | FP8/INT4/GPTQ/AWQ/GGUF/NVFP4/Marlin/Machete |
| 13 | distributed | TP/PP/DP/EP 并行策略、通信原语、Elastic EP |
| 14 | lora | LoRA Layer 实现、Punica Wrapper、动态加载卸载 |
| 15 | api_layer | OpenAI/Anthropic/gRPC API、Batch Serving、MCP Tool Server |
| 16 | cuda_kernels | 注意力/缓存/激活/量化/MoE/通信/采样内核 |
| 17 | compilation_and_optimization | torch.compile、CUDA Graphs、Partition Rules |
| 18 | structured_output | xgrammar/outlines/lm-format-enforcer/guidance 后端 |
| 19 | speculative_decode | EAGLE/Medusa/N-gram/DFlash/Suffix Decoding 方法 |
| 20 | metrics_and_monitoring | Prometheus 指标、日志系统、OpenTelemetry、Profiling |
| 21 | design_patterns | 9 种设计模式清单、可扩展性设计、性能优化策略 |

#### Scenario: 单个分模块文件质量标准
- **WHEN** 用户打开任意分模块文件
- **THEN** 文件应包含：(1) 定位说明 + 📌 Mermaid 图示化总括 (2) 核心概念解释 (3) Mermaid 架构图 (4) 来自实际源码的代码片段（标注文件路径和行号）(5) 数据流描述 (6) 设计决策理由

### Requirement: 文档质量约束
所有文档 SHALL 遵循以下质量标准：
- 中文撰写，技术术语保留英文
- 先宏观后细化：每节先讲"是什么、为什么"，再讲"怎么实现"
- 面向系统架构师：侧重设计权衡、接口契约、扩展机制
- 每个 Mermaid 图使用不同配色方案便于区分
- 所有代码引用均可通过文件路径在 `/workspace/vllm` 源码中定位
- 基于 v1 架构为主进行分析（v0 仅在演进部分提及）
