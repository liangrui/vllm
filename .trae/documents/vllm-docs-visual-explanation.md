# 计划：vLLM 使用文档详细解读与可视化图表讲解

## 目标
创建一份全面的 vLLM 使用文档解读文档，通过 Mermaid 可视化图表深入讲解 vLLM 的架构、核心概念、功能特性、部署方式和使用方法。

## 输出文件
`/workspace/docs_explained.md` — 一份包含 Mermaid 图表的中文解读文档

## 实施步骤

### 步骤 1：文档总览与导航结构
- 绘制 vLLM 文档的整体导航结构图（Mermaid mindmap）
- 说明文档按用户角色（使用者/开发者/贡献者）的组织方式
- 包含 `.nav.yml` 中定义的导航层级

### 步骤 2：架构总览（Architecture Overview）
- 绘制 vLLM V1 多进程架构图（API Server → Engine Core → GPU Worker → DP Coordinator）
- 绘制入口点关系图（LLM 类 vs OpenAI API Server vs AsyncLLMEngine）
- 绘制类层次结构图（VllmConfig → Engine → Worker → ModelRunner → Model）
- 解释进程数量计算公式：`A + DP + N (+1 if DP>1)`

### 步骤 3：核心机制 — PagedAttention
- 绘制 PagedAttention 的 KV Cache 分块存储示意图
- 绘制 Q/K/V 计算流程图（Query → Key → QK → Softmax → Value → LV → Output）
- 解释 Block、Thread Group、Warp、Grid 等概念
- 对比传统连续内存 vs 分页内存管理

### 步骤 4：前缀缓存（Prefix Caching）
- 绘制基于哈希的前缀缓存机制流程图
- 绘制 Block 分配/释放/驱逐的时序图
- 展示多请求共享前缀的缓存命中示例
- 解释 KVCacheBlock 数据结构和双向链表自由队列

### 步骤 5：CUDA Graphs
- 绘制 CUDA Graphs 模式切换图（NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE）
- 绘制 CudagraphDispatcher 调度流程图
- 绘制嵌套 Wrapper 设计图（FULL 外层 + PIECEWISE 内层）
- 展示 BatchDescriptor 调度键结构

### 步骤 6：功能特性矩阵
- 绘制 Feature × Feature 兼容性热力图
- 绘制 Feature × Hardware 兼容性热力图
- 量化方案对比表（FP8/INT8/INT4/AWQ/GPTQ/GGUF 等 14 种）
- 推测解码方法选择流程图（EAGLE/MTP/Draft Model/N-gram/Suffix）

### 步骤 7：部署方式
- 绘制部署架构选择决策树（Docker / K8s / 框架集成）
- 绘制 Data Parallel 部署架构图（内部负载均衡 vs 外部负载均衡 vs 混合负载均衡）
- 绘制分离式预填充（Disaggregated Prefilling）架构图
- 展示 Connector 类型及其选择指南

### 步骤 8：服务模式
- 绘制离线推理 vs 在线服务对比图
- 绘制 OpenAI 兼容 API 端点全景图（Completions / Chat / Embeddings / Transcriptions / Responses / Realtime 等）
- 绘制请求处理流水线（HTTP Request → Tokenization → Scheduling → Model Execution → Output Processing）

### 步骤 9：配置与优化
- 绘制 EngineArgs 配置层级图
- 绘制内存优化策略选择流程图
- 展示并行策略组合（TP × PP × DP × EP × CP）决策指南

### 步骤 10：训练集成
- 绘制 RLHF 训练集成架构图
- 绘制权重传输方式对比（IPC vs NCCL）
- 展示逐层训练（Layerwise）流程

## 技术方案
- 所有图表使用 Mermaid 语法（支持 flowchart、sequenceDiagram、classDiagram、mindmap 等）
- 文档使用中文撰写
- 每个章节包含：概念解释 → 可视化图表 → 关键代码/命令示例 → 注意事项
