# vLLM 代码深度分析报告 — 实施计划

## 一、概述

### 目标
对 `/workspace/vllm` 源码进行全面深度分析，产出一份**先宏观后细化**的高质量技术文档，面向系统架构师，包含丰富代码示例。

### 输出位置
`/workspace/ReadCode/` 目录

### 输出结构（总览 + 分模块详情）

```
/workspace/ReadCode/
├── vllm_code_structure_analysis.md          # 总览文件（更新已有文件）
├── 01_architecture_overview.md              # 架构总览：分层设计、版本演进、设计哲学
├── 02_config_system.md                      # 配置系统：VllmConfig 及所有子配置
├── 03_engine_core.md                        # 引擎核心：LLMEngine/AsyncLLM/EngineCore
├── 04_scheduler.md                          # 调度器：调度策略、请求队列、Continuous Batching
├── 05_pagedattention.md                     # PagedAttention：KV缓存管理、Block分配
├── 06_attention_backends.md                 # 注意力后端：FlashAttention/FlashInfer/MLA/Triton等
├── 07_model_executor.md                     # 模型执行器：ModelRunner、模型注册表、200+模型支持
├── 08_worker_and_executor.md                # Worker与执行器：UniProc/MultiProc/Ray Executor
├── 09_kv_cache.md                           # KV缓存系统：缓存协调器、offload、transfer
├── 10_sampling.md                           # 采样系统：采样参数、并行采样、推测解码
├── 11_multimodal.md                         # 多模态：图像/音频/视频处理管线
├── 12_quantization.md                       # 量化支持：FP8/INT4/GPTQ/AWQ/GGUF等
├── 13_distributed.md                        # 分布式：并行状态、通信原语、弹性扩展
├── 14_lora.md                               # LoRA：动态适配器加载与管理
├── 15_api_layer.md                          # API层：OpenAI/Anthropic/gRPC/Batch接口
├── 16_cuda_kernels.md                       # CUDA/C++内核：注意力/量化/MoE/采样内核
├── 17_compilation_and_optimization.md       # 编译优化：torch.compile/CUDA Graphs/Inductor
├── 18_structured_output.md                  # 结构化输出：outlines/xgrammar/lm-format-enforcer
├── 19_speculative_decode.md                 # 推测解码：Medusa/EAGLE/N-gram/DFlash
├── 20_metrics_and_monitoring.md             # 监控指标：Prometheus/日志/性能统计
└── 21_design_patterns.md                    # 设计模式汇总与架构洞察
```

---

## 二、各文件详细内容规划

### 文件0: `vllm_code_structure_analysis.md`（总览文件）
**定位**: 更新已有文件，作为整个分析的导航入口和宏观概览

**内容大纲**:
1. **项目概述** — 基本信息、核心价值主张、生态地位
2. **整体架构图** — 六层架构（服务层→引擎层→调度核心→执行器层→模型执行器→内核层），含 Mermaid 图
3. **版本演进路线** — v0 → v1 迁移策略、兼容性设计
4. **目录结构全景图** — 完整的目录树及每个目录的职责说明
5. **快速导航索引** — 链接到所有分模块详情文件
6. **关键技术特性一览表** — 每个特性的位置、原理、影响一句话总结
7. **数据流总览** — 从请求到响应的完整链路
8. **设计哲学总结** — 核心设计决策及其理由

---

### 文件1: `01_architecture_overview.md`
**定位**: 架构层面的深度分析，建立全局认知

**内容**:
- **分层架构详解**
  - 每一层的职责边界、接口定义、依赖关系
  - 层间通信机制（Protocol/Request/Output 数据结构）
  - 为什么这样分层（权衡分析）
- **v0 → v1 架构演进**
  - `engine/llm_engine.py` 如何包装 v1 版本（附代码）
  - v1 的核心改进点：解耦 EngineCore、异步调度优化
  - 向后兼容策略（`__getattr__` 延迟导入模式）
- **核心数据结构**
  - `Sequence` / `SequenceGroup` / `SequenceStatus`
  - `EngineInput` → `EngineCoreRequest` 转换链
  - `SamplerOutput` / `RequestOutput` 输出模型
- **设计原则**
  - 配置驱动（28+配置模块）
  - 注册表模式（模型/处理器/tokenizer）
  - 策略模式（注意力/线性层/量化后端选择）
  - 工厂模式（Executor/Worker创建）

---

### 文件2: `02_config_system.md`
**定位**: 完整配置体系分析

**内容**:
- **VllmConfig 主配置类**（完整字段列表+说明）
  - 读取 `config/vllm.py` 中所有子配置引用
- **各子配置详解**（每个配置类的关键字段和默认值）:
  - `ModelConfig` — 模型架构、dtype、tokenizer
  - `CacheConfig` — KV缓存块大小、GPU内存比例
  - `ParallelConfig` — TP/PP/DP 并行度
  - `SchedulerConfig` — 调度策略、延迟阈值
  - `DeviceConfig` — 设备检测与选择
  - `LoadConfig` — 权重加载方式
  - `LoRAConfig` — LoRA适配器配置
  - `QuantizationConfig` — 量化方案选择
  - `CompilationConfig` — torch.compile / CUDA Graphs
  - `SpeculativeConfig` — 推测解码参数
  - `ObservabilityConfig` — 可观测性设置
  - 其余配置（Offload/KVTransfer/SpeechToText/Reasoning 等）
- **配置验证与合并逻辑**
- **环境变量覆盖机制** (`env_override.py`)
- **OptimizationLevel (O0-O3)** 各级别含义

---

### 文件3: `03_engine_core.md`
**定位**: 引擎层的核心实现分析

**内容**:
- **LLMEngine 类** (`v1/engine/llm_engine.py`)
  - 初始化流程：配置解析 → Executor选择 → Worker创建
  - `add_request()` / `step()` / `abort_request()` 核心方法
  - 与 EngineCoreClient 的交互
- **AsyncLLM** (`v1/engine/async_llm.py`)
  - 异步引擎实现
  - LLM 高级 API 封装（`generate()` / `chat()`）
- **EngineCore** (`v1/engine/core.py`)
  - 解耦后的核心执行循环
  - InputProcessor → Scheduler → Worker → OutputProcessor 管线
- **InputProcessor** — 请求预处理（tokenization、multimodal处理）
- **OutputProcessor** — 输出后处理（detokenization、logprobs计算）
- **Detokenizer** — 异步反 tokenization
- **Coordinator** — 多节点协调
- **异常处理机制**

---

### 文件4: `04_scheduler.md`
**定位**: 调度系统的深度剖析

**内容**:
- **Scheduler 类** (`v1/core/sched/scheduler.py`)
  - `schedule()` 方法完整逻辑：
    - 请求队列管理（waiting/running/swapped/finished）
    - prefill/decode 请求选择策略
    - KV 缓存块分配算法
    - 输出 SchedulerOutput
- **RequestQueue** — 优先级队列、抢占机制
- **Continuous Batching 原理**
  - 对比静态批处理的优劣
  - Chunked Prefill 实现
  - Prefill-Decode 解耦调度
- **调度策略细节**
  - FCFS (First-Come-First-Served)
  - 双策略调度（prefill优先 vs decode优先）
  - 请求抢占（preemption）与恢复
- **SchedulerOutput 数据结构**
- **AsyncScheduler** — 异步调度变体

---

### 文件5: `05_pagedattention.md`
**定位**: PagedAttention 核心创新深度分析

**内容**:
- **问题背景**: 传统 KV 缓存的内存浪费问题
  - 固定分配导致的内部/外部碎片
  - 内存利用率低（通常 < 50%）
- **PagedAttention 设计思想**
  - 操作系统虚拟内存类比
  - Block（页）的概念：固定大小（16 tokens）
  - 逻辑地址 → 物理地址映射（BlockTable）
  - Block Pool（物理内存池）
- **关键实现**:
  - `KVCacheManager` (`v1/core/kv_cache_manager.py`)
    - `allocate()` / `free()` / `can_allocate()`
    - Block 分配与回收策略
  - `BlockPool` (`v1/core/block_pool.py`)
    - 内存池管理
  - `SingleTypeKVCacheManager` — 单类型缓存管理
- **内存共享机制**（Memory Sharing）
  - Copy-on-Write
  - Beam Search 共享
  - Parallel Sampling 共享
- **Prefix Caching** — 公共前缀自动识别与复用
- **实际效果数据**: 2-4x 内存节省

---

### 文件6: `06_attention_backends.md`
**定位**: 所有注意力后端实现的详细分析

**内容**:
- **注意力后端选择器** (`v1/attention/selector.py`)
  - 选择逻辑：FlashAttention → FlashInfer → ROCm → Triton → CPU
- **FlashAttention 后端** (`backends/flash_attn.py`)
  - FlashAttention-2 / FlashAttention-3 集成
  - 不同 KV 长度的处理（prefill vs decode）
- **FlashInfer 后端** (`backends/flashinfer.py`)
  - FlashInfer 库集成
  - Page-level 注意力操作
- **MLA (Multi-head Latent Attention)** (`backends/mla/`)
  - DeepSeek-V3/V4 专用
  - KV 压缩 + 稀疏注意力
  - 多种 MLA 变体（flash_attn/flashinfer/triton/cutlass/rocm_aiter）
- **Triton 注意力** (`backends/triton_attn.py`)
  - 自定义 Triton kernel
  - Paged Attention 的 Triton 实现
- **ROCm 后端** — AMD GPU 支持
- **CPU 注意力** — CPU fallback
- **其他特殊注意力**:
  - Mamba/Mamba2 注意力（SSM 替代）
  - Linear Attention
  - Short Conv Attention
  - GDN (Generalized Dynamic Navie) Attention
- **注意力操作层** (`ops/`) — 底层算子:
  - `paged_attn.py` — 分页注意力核心
  - `chunked_prefill_paged_decode.py` — chunked prefill + paged decode
  - `triton_prefill_attention.py` / `triton_decode_attention.py`

---

### 文件7: `07_model_executor.md`
**定位**: 模型执行层全貌

**内容**:
- **ModelRegistry** — 模型注册表机制
  - 装饰器注册模式
  - 200+ 支持的模型架构列表（按家族分类）
- **模型接口定义** (`interfaces.py` / `interfaces_base.py`)
  - `ModelForCausalLM` 基类
  - forward() 签名约定
- **主要模型家族分析**（选取代表性模型深入）:
  - **LLaMA 系列** (`llama.py`) — 最广泛支持的架构
  - **DeepSeek V2/V3/V4** — MoE + MLA
  - **Qwen2/Qwen3/Qwen3.5** — 国产大模型
  - **Mistral/Gemma** — 开源主流
  - **MoE 模型** (Mixtral/DeepSeek-QExa) — 专家混合
  - **多模态模型** (LLaVA/Qwen-VL) — 视觉语言模型
- **Transformer 层组件**:
  - Attention 层 (`layers/attention/`)
  - MLP/FeedForward 层
  - RMSNorm / LayerNorm
  - RoPE 位置编码
- **自定义操作注册** (`custom_op.py`) — torch.library 集成

---

### 文件8: `08_worker_and_executor.md`
**定位**: 执行框架分析

**内容**:
- **Executor 抽象** (`v1/executor/abstract.py`)
  - 统一执行接口
- **UniProcExecutor** — 单进程执行
  - 适用场景：单 GPU
- **MultiProcExecutor** — 多进程执行
  - Python multiprocessing
  - 进程间通信
- **RayExecutorV2** — Ray 分布式执行
  - Ray 集群初始化
  - Actor 管理
  - 资源调度
- **WorkerBase** — Worker 抽象基类
  - `execute_model()` / `initialize_profile()` / `load_model()`
- **GPUWorker** (`gpu_worker.py`)
  - GPU 模型推理执行
  - CUDA 流管理
- **GPUModelRunner** (`gpu_model_runner.py`)
  - 模型前向传播编排
  - Sampling 执行
  - KV Cache 写入
- **CPUWorker / CPUModelRunner** — CPU 执行路径
- **TPUWorker / XPUWorker** — 其他硬件
- **输入批处理** (`gpu_input_batch.py`, `ubatching.py`)
  - micro-batching 策略
  - padding 优化

---

### 文件9: `09_kv_cache.md`
**定位**: KV 缓存系统完整分析

**内容**:
- **KV Cache Manager 层次**
  - `KVCacheManager` — 顶层抽象
  - `SingleTypeKVCacheManager` — 单类型实现
  - `EncoderCacheManager` — 编码器缓存
- **KV Cache Coordinator** — 多 GPU 缓存协调
- **KV Cache 接口** (`kv_cache_interface.py`)
  - 统一的缓存访问 API
- **KV Cache Metrics** — 缓存使用统计
- **KV Offload** (`v1/kv_offload/`, `v1/simple_kv_offload/`)
  - CPU 卸载策略
  - 复用管理器
- **KV Transfer** (`distributed/kv_transfer/`)
  - 跨节点 KV 缓存传输
  - Disaggregated Prefill 工作流
- **Block Table 管理** (`block_table.py`)
- **前缀缓存 (Prefix Caching)**
  - 自动公共前缀检测
  - 缓存命中/未命中处理
  - 与调度器的集成

---

### 文件10: `10_sampling.md`
**定位**: 采样与生成策略分析

**内容**:
- **SamplingParams** (`sampling_params.py`)
  - temperature / top_p / top_k / max_tokens
  - presence_penalty / frequency_penalty
  - repetition_penalty / min_p
  - stop sequences / skip_special_tokens
- **Sampler** (`v1/sample/sampler.py`)
  - `sample()` 核心方法
  - logits processor 链
  - 概率分布变换
- **RejectionSampler** (`rejection_sampler.py`)
  - 推测解码验证
- **ParallelSampling** (`parallel_sampling.py`)
  - 并行采样（beam search 等）
  - ParentRequest 管理
- **Logprobs 计算** (`logprobs.py`)
  - token 概率追踪
- **Thinking Budget State** — 推理预算控制
- **Speculative Decode Metadata** — 推测解码元数据

---

### 文件11: `11_multimodal.md`
**定位**: 多模态处理管线分析

**内容**:
- **MultiModalRegistry** — 多模态处理器注册表
- **媒体连接器** (`media/`)
  - Image Connector — 图像加载与预处理
  - Audio Connector — 音频处理
  - Video Connector — 视频帧提取
- **处理器** (`processing/`)
  - Processor 基类
  - 上下文注入
  - Dummy inputs 生成
- **输入解析** (`parse.py`) — 多模态输入识别
- **特征提取管线**
  - CLIP / SigLIP 视觉编码器
  - Whisper / 音频编码器
  - Encoder Budget Management — 控制 token 数量
- **支持的多模态模型**
  - LLaVA-Next / Qwen2-VL / Pixtral 等
- **Assets 系统** (`assets/`) — 多模态资源管理

---

### 文件12: `12_quantization.md`
**定位**: 量化方案全面分析

**内容**:
- **量化方案矩阵**（精度/权重/激活/用途对比表格）
- **FP8 量化**
  - H100/H200 原生 FP8 支持
  - 动态量化策略
  - MXFP8 / MXFP4 块缩放浮点
- **INT8 / INT4 量化**
  - 量化配置类层次
  - GPTQ — 训练后量化
  - AWQ — 激活感知量化
- **GGUF 格式** — llama.cpp 兼容
- **NVFP4** — Blackwell 4-bit 浮点
- **Marlin / Machete** — 高性能 4-bit 内核
- **量化线性层** (`kernels/linear/`) — 量化 GEMM
- **量化配置** (`layers/quantization/`) — 各方案的配置类
- **CUDA 量化内核** (`csrc/quantization/`) — AWQ/GPTQ/GGUF/Marlin/Machete

---

### 文件13: `13_distributed.md`
**定位**: 分布式计算分析

**内容**:
- **并行策略**
  - Tensor Parallelism (TP) — 张量并行
  - Pipeline Parallelism (PP) — 流水线并行
  - Data Parallelism (DP) — 数据并行
  - Expert Parallelism (EP) — 专家并行（MoE）
- **并行状态管理** (`parallel_state.py`)
  - TP/PP/DP/EP 组初始化
  - NCCL 通信域
- **设备通信器** (`device_communicators/`)
  - AllReduce / AllGather / ReduceScatter
  - Custom AllReduce — 自定义归约
  - Quick AllReduce — 快速归约
  - PYNCL — NCCL 包装
  - SHM Broadcast — 共享内存广播
- **KV Transfer** — 跨节点缓存传输
- **Weight Transfer** — 权重分发（NCCL / IPC）
- **EC Transfer** — 纠删码传输
- **Elastic EP** — 弹性专家并行
- **EPLB** — 弹性负载均衡
- **Stateless Coordinator** — 无状态协调器
- **NIXL Utils** — NIXL 互操作

---

### 文件14: `14_lora.md`
**定位**: LoRA 适配器系统分析

**内容**:
- **LoRA 概述** — Low-Rank Adaptation 原理
- **LoRAConfig** — 配置参数
- **LoRA Model** (`lora_model.py`) — LoRA 模型封装
- **LoRA Weights** (`lora_weights.py`) — 权重管理
- **LoRA Layer 实现** (`layers/`)
  - ColumnParallelLinear + LoRA
  - RowParallelLinear + LoRA
  - FusedMoE + LoRA
  - LogitsProcessor + LoRA
- **Punica Wrapper** (`punica_wrapper/`)
  — GPU LoRA 计算库集成
  - CPU/XPU 变体
- **LoRA Request** — 请求级 LoRA 绑定
- **LoRA Resolver** — 适配器解析（FileSystem/HF Hub）
- **WorkerManager** — Worker 级 LoRA 管理
- **动态加载/卸载** — 运行时适配器切换

---

### 文件15: `15_api_layer.md`
**定位**: 服务接口层分析

**内容**:
- **OpenAI 兼容 API** (`entrypoints/openai/`)
  - `api_server.py` — FastAPI 服务器
  - 路由定义：/v1/chat/completions / /v1/completions / /v1/embeddings
  - 流式响应（SSE）
  - Orca Metrics
- **Anthropic API** (`entrypoints/anthropic/`)
  - 协议适配
  - 消息格式转换
- **gRPC Server** (`grpc_server.py`)
- **CLI 入口** (`entrypoints/cli/`)
  - `serve` / `run_batch` / `collect_env` / `openai` 命令
- **Batch Serving** — 离线批处理
- **SageMaker 集成**
- **MCP Tool Server** — 模型上下文协议工具服务
- **Pooling 端点** — 嵌入池化接口
- **LLM 高级 API** (`entrypoints/llm.py`)
  - `generate()` / `encode()` / `chat()`
- **SSL/TLS 支持**
- **请求日志与访问过滤**

---

### 文件16: `16_cuda_kernels.md`
**定位**: CUDA/C++ 内核层深度分析

**内容**:
- **构建系统**
  - CMakeLists.txt 结构
  - torch_utils 扩展编译
- **注意力内核** (`csrc/attention/`)
  - `paged_attention_v1.cu` / `paged_attention_v2.cu`
  - MLA 相关内核
  - 合并注意力状态
- **缓存内核**
  - `cache_kernels.cu` — KV 读写
  - `reshape_and_cache` — 重塑缓存
- **激活函数内核**
  - Gelu / SiLU / ReLU
  - RMSNorm / LayerNorm
  - RoPE 位置编码
- **量化内核** (`csrc/quantization/`)
  - AWQ 反量化 + GEMM
  - GPTQ 量化 GEMM
  - GGUF 反量化
  - Marlin 4-bit
  - Machete 新一代量化
  - CUTLASS 扩展（W8A8, FP4, NVFP4）
- **MoE 内核** (`csrc/moe/`)
  - Mixtral-style MoE
  - TopK 路由
  - WNA16 (4-bit MoE)
  - DSv3 路由
  - MXFP8 MoE
- **通信内核**
  - Custom AllReduce
  - Custom Quick Reduce
  - CU Memory Allocator
- **采样内核**
  - Top-k / Top-p 采样
  - 持久化 TopK
- **平台特定**
  - CPU 内核 (AMX, NEON, VSX, VXE)
  - ROCm 内核 (AMD GPU)

---

### 文件17: `17_compilation_and_optimization.md`
**定位**: 编译与运行时优化分析

**内容**:
- **torch.compile 集成**
  - Dynamo + Inductor 编译流水线
  - 自定义 op 注册（`torch.library.custom_op`）
  - 编译 passes (`compilation/passes/`)
    - Inductor pass
    - vLLM inductor pass
    - Pass manager
- **CUDA Graphs** (`compilation/cuda_graph.py`)
  - Piecewise CUDAGraphs — 子图捕获（推荐）
  - Full CUDAGraphs — 整体捕获
  - CUDAGraph dispatcher
  - Encoder CUDA Graphs
- **Partition Rules** — 图分割规则
- **Codegen** — 代码生成
- **Caching** — 编译缓存
- **Monitor** — 编译监控
- **Counter** — 性能计数
- **BaseStaticGraph** — 静态图基类
- **PiecewiseBackend** — 分段后端
- **Triton Utils** — Triton 编译工具
  - JIT Monitor
  - Memory Allocation
- **OptimizationLevel O0-O3** 含义详解

---

### 文件18: `18_structured_output.md`
**定位**: 结构化输出约束分析

**内容**:
- **StructuredOutputsConfig** — 配置
- **后端实现**:
  - **xgrammar** (`backend_xgrammar.py`) — 高性能语法约束
  - **outlines** (`backend_outlines.py`) — 基于 FSM 的约束
  - **lm-format-enforcer** (`backend_lm_format_enforcer.py`) — 轻量级格式强制
  - **guidance** (`backend_guidance.py`) — Guidance 库集成
- **支持的结构类型**
  - JSON Schema
  - 正则表达式
  - Grammar (EBNF/CFG)
  - Choice 约束
  - 工具调用格式
- **Request 级别配置**
- **工具函数**

---

### 文件19: `19_speculative_decode.md`
**定位**: 推测解码加速技术分析

**内容**:
- **推测解码原理**
  - Draft Model + Target Model 架构
  - 验证阶段（Rejection Sampling）
  - 典型加速比 2-4x
- **SpeculativeConfig** — 配置
- **Draft Model 接口** (`draft_model.py`)
- **推测解码方法**:
  - **EAGLE** (`eagle.py`) — Early Exit Layer
  - **Medusa** (`medusa.py`) — 多头预测
  - **N-gram Proposer** (`ngram_proposer.py`) — N-gram 草稿
  - **N-gram GPU Proposer** — GPU 加速 N-gram
  - **DFlash** (`dflash.py`) — DFlash 推测解码
  - **Suffix Decoding** (`suffix_decoding.py`) — 后缀解码
  - **Gemma4** (`gemma4.py`) — Gemma4 专用
  - **MLP Speculator** — MLP 草稿模型
- **LLM Base Proposer** — 基于 LLM 的草稿
- **Extract Hidden States** — 隐藏状态提取
- **Metrics** — 推测解码指标（接受率、加速比）

---

### 文件20: `20_metrics_and_monitoring.md`
**定位**: 可观测性分析

**内容**:
- **指标收集** (`v1/metrics/`)
  - `perf.py` — 性能指标（延迟 TTFT/TBT/TPL, 吞吐量, GPU 利用率）
  - `stats.py` — 统计数据（迭代统计、前缀缓存统计）
  - `prometheus.py` — Prometheus 导出端点
  - `reader.py` — 指标读取接口
  - `utils.py` — 指标工具
- **日志系统**
  - `loggers.py` — 日志工厂
  - `ray_wrappers.py` — Ray 日志适配
- **StatLoggerFactory / StatLoggerManager**
- **OpenTelemetry 集成** (`tracing/otel.py`)
- **Profiling**
  - `layerwise_profile.py` — 逐层性能分析
  - NVTX hooks
- **Usage Context & Usage Lib**
- **Observability Config**

---

### 文件21: `21_design_patterns.md`
**定位**: 设计模式与架构洞察总结

**内容**:
- **设计模式清单**
  - 策略模式（注意力/线性/量化后端）
  - 工厂模式（Executor/Worker/Connector）
  - 观察者模式（指标收集/事件发布）
  - 注册表模式（模型/多模态/tokenizer/渲染器）
  - 装饰器模式（模型注册/配置验证）
  - 适配器模式（API协议转换）
  - 建造者模式（配置构建）
  - 原型模式（请求复制）
- **代码组织原则**
  - 分层职责分离
  - 接口抽象（Protocol/ABC）
  - 配置驱动开发
  - 错误处理层次
- **架构优势与权衡**
- **可扩展性设计**
  - 如何添加新模型
  - 如何添加新硬件后端
  - 如何添加新量化方案
- **性能优化策略总结**
- **与竞品对比视角（架构层面）**
- **未来发展方向推断**

---

## 三、实施步骤

### Step 1: 更新总览文件 `vllm_code_structure_analysis.md`
- 重写为导航入口 + 宏观概览
- 包含完整的文件索引链接
- 绘制架构图和数据流图

### Step 2: 按顺序创建 01-21 号分模块文件
- 每个文件遵循统一模板：概述 → 核心概念 → 关键实现（带代码）→ 数据流 → 设计决策
- 代码片段从实际源码中提取，附行号引用和注释
- 先宏观架构描述，再深入实现细节

### Step 3: 交叉引用与一致性检查
- 确保文件间交叉引用正确
- 统一术语和格式风格
- 检查代码引用准确性

## 四、质量标准

1. **每个文件** 都包含：概念解释、架构图(Mermaid)、核心代码（带注释）、数据流、设计决策理由
2. **代码示例** 均来自实际源码，标注文件路径和行号范围
3. **先宏观后细化**：每节先讲"是什么、为什么"，再讲"怎么实现"
4. **面向系统架构师**：侧重设计权衡、接口契约、扩展机制，而非逐行代码走读
5. **中文撰写**，技术术语保留英文

## 五、假设与决策

1. 分析基于 `/workspace/vllm` 当前 main 分支最新代码
2. 以 v1 架构为主进行分析（v0 仅在演进部分提及）
3. 不覆盖测试代码（tests/）和基准测试（benchmarks/）
4. 文档版本标记为 2.0（相比已有的 1.0 版本大幅扩展）
5. 每个 Markdown 文件独立可读，同时通过总览文件串联

## 六、验证方式

1. 所有文件中的代码引用均可通过文件路径在源码中定位
2. 总览文件的索引链接指向正确的分模块文件
3. Mermaid 图表语法正确可渲染
