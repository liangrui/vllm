# MPT（MosaicML Pretrained Transformer）原理详解

> 本文结合 MPT 论文与 vLLM 源码（`vllm/model_executor/models/mpt.py`）系统阐述 MPT 的架构原理、前向流程与代码实现，并配以具体数值示例帮助理解。

---

## 一、背景与论文出处

MPT（MosaicML Pretrained Transformer）是 MosaicML（2023 年 6 月被 Databricks 收购）于 2023 年 5–6 月发布的开源大语言模型系列，代表作包括 MPT-7B、MPT-30B、MPT-7B-Instruct、MPT-7B-Chat、MPT-7B-StoryWriter-65k+。

MPT 系列并没有发表单独的"理论论文"，其架构描述以 **官方博客 + HuggingFace 模型卡 + `llm-foundry` 训练代码** 的形式给出，并大量借鉴以下两篇核心论文：

1. **ALiBi（Attention with Linear Biases）**
   - Press, R. et al., *"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"*, arXiv:2108.12409.
   - 论文核心：用线性距离偏置取代位置编码，使得模型在训练时未见过的更长序列上仍可外推。

2. **FlashAttention**
   - Dao, T. et al., *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022.
   - 让长上下文训练可行，MPT 默认使用 FlashAttention 内核。

MPT 自身的工程贡献在于：**把这些组件拼装为一个稳定、可大规模训练、商业可用的开源模型族**。它在 LLaMA-7B 之上做了若干"工程化改良"——ALiBi 替代 RoPE / 学习式位置编码、QK LayerNorm、可选的 QKV clip、Low-Precision LayerNorm、删除 bias、Tied Embedding 等，使训练在 1T token、9.5 天、440×A100 上稳定收敛。

---

## 二、整体架构概览

MPT 是一个标准的 **Decoder-Only Transformer**，结构上接近 GPT-NeoX / LLaMA，但在以下几处有差异：

| 组件 | GPT-2 / LLaMA | MPT |
|------|---------------|-----|
| 位置编码 | 学习式 / RoPE | **ALiBi（无位置嵌入）** |
| LayerNorm | 标准 / RMSNorm | **Low-Precision LayerNorm** |
| QK 是否归一化 | 无 | **QK LayerNorm（可选，默认开）** |
| Linear 是否带 bias | 是 | **不带 bias（`no_bias=True`）** |
| 激活函数 | GeLU / SwiGLU | **GELU** |
| 词嵌入与输出 | 解耦 | **Tied（共享权重）** |
| 注意力实现 | 标准 / SDPA | **FlashAttention（带 ALiBi 偏置）** |
| 训练稳定性 | – | QKV clip + LayerNorm + ALiBi 共同保证 |

MPT-7B 的关键超参：

```
n_layers          : 32
n_heads           : 32
d_model           : 4096
vocab_size        : 50432
max_seq_len       : 2048（可外推到 65k+，StoryWriter 变体）
expansion_ratio   : 4
no_bias           : True
alibi_bias_max    : 8
qk_ln             : True
clip_qkv          : None（base 模型默认无）
```

---

## 三、核心原理：ALiBi（Attention with Linear Biases）

### 3.1 标准 Attention 的位置编码痛点

传统 Transformer 把位置编码（正弦/学习式/RoPE）**加到 token embedding 上**，再做 `Q·Kᵀ`。这类方法存在两个问题：

1. 训练时见过的最大长度 `L_train` 之外，位置编码进入未训练区域，性能急剧下降（不可外推）。
2. 增大上下文窗口需要重新训练或微调。

### 3.2 ALiBi 的核心思想

ALiBi **不在输入端注入位置信息**，而是在计算 attention score 时，**直接给注意力分数矩阵加上一个线性偏置**：

$$
\text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}} \;-\; m_h \cdot |i - j|\right)
$$

- `i, j` 分别是 query 和 key 的位置下标。
- `m_h` 是 **每个注意力头独有的常数斜率**（不学习，固定）。
- 距离越远的 (key, query) 对，attention score 被惩罚得越多，使模型天然偏向"近处 token"。
- 对因果掩码（causal mask）位置 (j > i)，原本就是 `-∞`，再叠加 `-m·(j-i)` 不改变。

### 3.3 斜率 m_h 的构造

ALiBi 论文给出斜率集合：当头数为 2 的幂时，使用

$$
m_h = \frac{1}{2^h}, \quad h = 1, 2, \dots, n
$$

也就是 `{1/2, 1/4, 1/8, 1/16, ...}` 几何序列。当头数不是 2 的幂时，取最近 2 的幂的几何序列再均匀截取。

vLLM 中的实现 `vllm/model_executor/models/mpt.py` 第 43–53 行 `mpt.py#L43-L53`：

```python
def _get_alibi_slopes(total_num_heads: int, alibi_bias_max: int) -> torch.Tensor:
    next_power_of_2 = 2 ** math.ceil(math.log2(total_num_heads))
    m = torch.arange(1, next_power_of_2 + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / next_power_of_2)
    slopes = 1.0 / torch.pow(2, m)
    if next_power_of_2 != total_num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:total_num_heads]
    return slopes
```

- `alibi_bias_max`（MPT 默认 8）控制最大斜率上限，乘到 `m` 上。
- 当 `total_num_heads` 不是 2 的幂时，先把几何序列按"奇偶交错重排"再截断，保证不同头斜率分布合理。

### 3.4 ALiBi 的外推性质

由于偏置只依赖于**位置距离 `|i - j|`** 而不依赖绝对位置，模型在推理时可以处理比训练时更长的序列——只要距离仍落在已"训练过的语义"内，attention 行为就稳定。这是 MPT-7B-StoryWriter 能在 65k+ 上下文推理的关键。

---

## 四、vLLM 中 MPT 的源码流程

源码文件：`vllm/model_executor/models/mpt.py`。整体类层次：

```
MPTForCausalLM (顶层)
├── transformer: MPTModel
│   ├── wte: VocabParallelEmbedding        # 词嵌入（与 lm_head 共享）
│   ├── blocks: List[MPTBlock]             # N 个 Transformer Block
│   │   ├── norm_1: LayerNorm
│   │   ├── attn: MPTAttention
│   │   │   ├── Wqkv: QKVParallelLinear
│   │   │   ├── q_ln / k_ln: LayerNorm     # QK LayerNorm（可选）
│   │   │   ├── out_proj: RowParallelLinear
│   │   │   └── attn: Attention            # 底层 FlashAttention + ALiBi
│   │   ├── norm_2: LayerNorm
│   │   └── ffn: MPTMLP
│   │       ├── up_proj: ColumnParallelLinear
│   │       ├── act: GELU
│   │       └── down_proj: RowParallelLinear
│   └── norm_f: LayerNorm
└── lm_head = transformer.wte              # 权重绑定
```

### 4.1 `MPTAttention`（`mpt.py#L56-L149`）

```python
class MPTAttention(nn.Module):
    def __init__(self, config, cache_config=None, quant_config=None, prefix=""):
        ...
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=not config.no_bias,
            ...
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(...)

        # 1) 张量并行切分 head
        self.num_heads = self.total_num_heads // tp_world_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)

        # 2) 取出当前 TP rank 对应 head 的 ALiBi 斜率
        alibi_slopes = _get_alibi_slopes(self.total_num_heads, self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        # 3) 把斜率传给 vLLM Attention，FlashAttention 内核会自动叠加 ALiBi 偏置
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            alibi_slopes=alibi_slopes,         # ← 关键
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, position_ids, hidden_states):
        del position_ids                       # ALiBi 不依赖 position_ids
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        attn_output = self.attn(q, k, v)        # ALiBi 偏置在底层 attention 内叠加
        output, _ = self.out_proj(attn_output)
        return output
```

要点：

- **`Wqkv` 融合 QKV 投影**：一次矩阵乘法得到 Q、K、V，节省访存，是 FlashAttention 友好的写法。
- **`clip_qkv`**：可选，对 QKV 数值范围做硬截断，进一步稳定训练（MPT-7B base 默认关闭）。
- **QK LayerNorm**：在 Q、K 进入 attention 之前各做一次 LayerNorm，提高数值稳定性，避免 QKᵀ 爆炸。
- **`del position_ids`**：MPT 使用 ALiBi 后，attention 模块本身不再需要 `position_ids`；但 vLLM 框架仍传入，这里显式忽略。
- **`alibi_slopes`**：每个 TP rank 只保留自己负责 head 的斜率，传给 `Attention`，由底层 FlashAttention 内核在 `softmax` 前向 attn score 减去 `m_h · |i - j|`。

### 4.2 `MPTMLP`（`mpt.py#L152-L183`）

```python
class MPTMLP(nn.Module):
    def __init__(self, config, ...):
        intermediate_size = config.expansion_ratio * hidden_size   # 4× hidden
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, ...)
        self.act = get_act_fn("gelu")
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, ...)

    def forward(self, x):
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x
```

标准 FFN：`up_proj → GELU → down_proj`，expansion_ratio=4。

### 4.3 `MPTBlock`（`mpt.py#L186-L217`）—— Pre-Norm 残差结构

```python
class MPTBlock(nn.Module):
    def __init__(self, config, ...):
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn   = MPTAttention(config, ...)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn    = MPTMLP(config, ...)

    def forward(self, position_ids, hidden_states):
        x = self.norm_1(hidden_states)            # Pre-Norm
        x = self.attn(position_ids=position_ids, hidden_states=x)
        hidden_states = hidden_states + x         # 残差

        x = self.norm_2(hidden_states)            # Pre-Norm
        x = self.ffn(x)
        hidden_states = hidden_states + x         # 残差
        return hidden_states
```

**Pre-Norm**（先归一化、再进子层），训练稳定性优于 Post-Norm，是大模型标配。

### 4.4 `MPTModel` 与 `MPTForCausalLM`

```python
@support_torch_compile
class MPTModel(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        ...
        assert config.embedding_fraction == 1.0
        assert config.norm_type == "low_precision_layernorm"
        self.wte = VocabParallelEmbedding(config.vocab_size, config.d_model)
        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_layers,
            lambda prefix: MPTBlock(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.blocks",
        )
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                    module.register_parameter("bias", None)   # 删除 bias

    def forward(self, input_ids, position_ids, intermediate_tensors=None, ...):
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = intermediate_tensors["hidden_states"]   # 流水线并行
        for block in islice(self.blocks, self.start_layer, self.end_layer):
            hidden_states = block(position_ids, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.norm_f(hidden_states)                 # 最终 LayerNorm
        return hidden_states


class MPTForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config, prefix=""):
        ...
        assert config.tie_word_embeddings
        self.transformer = MPTModel(vllm_config=vllm_config, prefix=...)
        self.lm_head = self.transformer.wte                # ← 权重绑定
        self.logits_processor = LogitsProcessor(config.vocab_size)
```

要点：

- **`embedding_fraction == 1.0`**：MPT 不限制 embedding 梯度更新的比例（一些早期 GPT 用 `embedding_fraction < 1` 防止 embedding 阶梯爆炸）。
- **`norm_type == "low_precision_layernorm"`**：MPT 自定义的低精度 LayerNorm，计算更快。
- **`no_bias` 删除 bias**：训练时所有 Linear、LayerNorm 的 bias 被清空，节省参数与推理成本。
- **`tie_word_embeddings`**：`lm_head` 与 `wte` 共享权重，参数量减少 `vocab_size × d_model`（MPT-7B 约 2 亿）。
- **支持流水线并行（PP）与张量并行（TP）**：通过 `get_pp_group()`、`make_layers` 与 `QKVParallelLinear` 等实现。

### 4.5 整体前向数据流

输入 `input_ids: [batch, seq]`，`position_ids: [batch, seq]`：

```
input_ids
   │
   ▼ VocabParallelEmbedding(wte)
hidden_states: [batch, seq, d_model]
   │
   ▼ for block in blocks:            # 32 层（MPT-7B）
   │     ┌──────────────────────────┐
   │     │ norm_1 (LayerNorm)        │
   │     │ Wqkv → [q,k,v]            │
   │     │ (clip_qkv?)               │
   │     │ q_ln, k_ln                │
   │     │ Attention(q,k,v,         │
   │     │   alibi_slopes=m_h)       │  ← ALiBi 在此叠加
   │     │ out_proj                  │
   │     │ + residual                │
   │     │ norm_2 (LayerNorm)        │
   │     │ up_proj → GELU → down_proj│
   │     │ + residual                │
   │     └──────────────────────────┘
   │
   ▼ norm_f (LayerNorm)
hidden_states: [batch, seq, d_model]
   │
   ▼ lm_head = wte (Tied)
logits: [batch, seq, vocab_size]
   │
   ▼ LogitsProcessor
   ▼ 采样 / argmax
next token
```

---

## 五、举例说明

### 例 1：ALiBi 斜率计算（MPT-7B，`alibi_bias_max = 8`）

`total_num_heads = 32`，正好是 2 的幂：

```
next_power_of_2 = 32
m = [1, 2, 3, ..., 32]                  # torch.arange(1, 33)
m = m * (8 / 32) = m * 0.25
  = [0.25, 0.5, 0.75, 1.0, 1.25, ..., 8.0]
slopes = 1 / 2^m
  = [2^-0.25, 2^-0.5, 2^-0.75, 2^-1, 2^-1.25, ..., 2^-8]
  ≈ [0.841, 0.707, 0.595, 0.500, 0.420, ..., 0.00391]
```

可见第 1 个头的斜率最大（≈0.84），关注最远的 token；第 32 个头的斜率最小（≈1/256），几乎只看当前位置——**不同头天然形成"远近注意力分工"**。

### 例 2：单头 ALiBi 偏置矩阵（4 个 token）

设斜率 `m_h = 0.5`，序列长度 `L = 4`。位置距离矩阵 `D[i, j] = |i - j|`：

```
        j=0  j=1  j=2  j=3
i=0  [  0    1    2    3 ]
i=1  [  1    0    1    2 ]
i=2  [  2    1    0    1 ]
i=3  [  3    2    1    0 ]
```

ALiBi 偏置 `-m_h · D`：

```
       j=0    j=1    j=2    j=3
i=0  [ 0.0  -0.5   -1.0   -1.5 ]
i=1  [-0.5   0.0   -0.5   -1.0 ]
i=2  [-1.0  -0.5    0.0   -0.5 ]
i=3  [-1.5  -1.0   -0.5    0.0 ]
```

叠加 causal mask 后（j>i 处为 `-∞`）：

```
       j=0    j=1    j=2    j=3
i=0  [ 0.0  -0.5   -1.0   -1.5 ]
i=1  [-0.5   0.0   -0.5   -1.0 ]
i=2  [-1.0  -0.5    0.0   -0.5 ]
i=3  [-1.5  -1.0   -0.5    0.0 ]
   （下三角已为 -∞，未画出）
```

观察：i=0 的 query 行，对自身得分为 0，越往后越被压制；i=3（最后一个 token）对最早 token（j=0）的偏置达到 -1.5，几乎不可能 attend 到——这正是 ALiBi "近因偏好"的本质。

### 例 3：MPT-7B 单层前向的形状流

假设 `batch = 2, seq = 16, d_model = 4096, n_heads = 32, head_dim = 128`：

| 阶段 | 张量形状 | 说明 |
|------|----------|------|
| `hidden_states` | `[2, 16, 4096]` | 输入 |
| `Wqkv(hidden_states)` | `[2, 16, 4096+4096+4096] = [2,16,12288]` | 融合 QKV |
| `q, k, v = split` | 各 `[2, 16, 4096]` | 切出 Q/K/V |
| `q.view → [2, 16, 32, 128]` | 多头 reshape |
| `Attention(q, k, v, alibi_slopes)` | `[2, 16, 4096]` | 内部叠加 ALiBi 偏置 + FlashAttention |
| `out_proj(attn_output)` | `[2, 16, 4096]` | 输出投影 |
| `+ residual` | `[2, 16, 4096]` | 残差连接 |
| `norm_2` | `[2, 16, 4096]` | Pre-Norm |
| `up_proj` | `[2, 16, 16384]` | FFN 上采样（4×） |
| `GELU` | `[2, 16, 16384]` | 激活 |
| `down_proj` | `[2, 16, 4096]` | FFN 下采样 |
| `+ residual` | `[2, 16, 4096]` | 残差连接 |

### 例 4：长度外推场景（训练 2K，推理 8K）

假设 MPT-7B 训练时 `max_seq_len = 2048`，推理时输入长度 `8192`：

- **传统位置编码模型**：8192 个位置中后半段没有训练过，对应的位置 embedding 处于未初始化状态，输出基本是噪声。
- **MPT + ALiBi**：偏置只依赖 `|i - j|`，最大距离也是 8192，但每个头的 `m_h` 已经在训练时把"远距离→低权重"的模式学会了。距离矩阵的形状没变（仍是三角矩阵），只是更"扁"——模型仍能生成合理输出，损失只会缓慢退化而非崩溃。这就是 MPT-7B-StoryWriter-65k+ 能在 65k 长上下文工作的根本原因。

### 例 5：用 vLLM 跑 MPT-7B 推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="mosaicml/mpt-7b", trust_remote_code=True)
prompts = ["The meaning of life is"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=64))
print(outputs[0].outputs[0].text)
```

vLLM 启动时会：

1. 读取 `config.json` → 解析 `attn_config.alibi / alibi_bias_max / qk_ln / clip_qkv`。
2. 实例化 `MPTForCausalLM` → `MPTModel` → `MPTBlock[]` → `MPTAttention`。
3. 在 `MPTAttention.__init__` 中调用 `_get_alibi_slopes(32, 8)` 得到 32 条斜率，按 TP rank 切分。
4. 把斜率传给底层 `Attention`，由 FlashAttention / FlashInfer 后端在 `softmax` 前叠加 ALiBi 偏置。
5. 加载权重：`wte` 与 `lm_head` 共享同一份 embedding；所有 Linear 的 `bias` 字段被显式置 `None`。
6. 推理时不需要传 `position_ids` 给 attention（`del position_ids`），但 vLLM 仍会维护位置以便 PagedAttention 索引 KV cache。

---

## 六、MPT 的工程价值小结

1. **训练稳定**：ALiBi + QK LayerNorm + (可选) QKV clip + Pre-Norm + Low-Precision LayerNorm 多重手段，使 7B/30B 模型在 1T token 上稳定收敛、无需人为干预。
2. **长上下文外推**：ALiBi 让模型可处理远超训练长度的输入，为 StoryWriter-65k+ 等变体提供基础。
3. **推理高效**：融合 QKV、`no_bias`、Tied Embedding、FlashAttention 让 vLLM 等推理框架能跑出高吞吐。
4. **商业友好**：Apache 2.0 + 全公开数据组成，是 2023 年开源 LLM 商业落地的标杆。
5. **vLLM 充分复用**：MPT 几乎所有计算都复用 vLLM 通用算子（`QKVParallelLinear`、`Attention`、`VocabParallelEmbedding`），TP / PP / 量化都"免费"支持，源码仅 335 行就完成集成。

---

## 七、参考

- Press, Smith, Lewis, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, arXiv:2108.12409.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022.
- MosaicML Blog, *Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs*, 2023-05-05.
- MosaicML Blog, *MPT-30B: Raising the bar for open-source foundation models*, 2023-06-22.
- vLLM 源码：[`vllm/model_executor/models/mpt.py`](file:///workspace/vllm/model_executor/models/mpt.py)
- vLLM 源码：[`vllm/model_executor/layers/attention/attention.py`](file:///workspace/vllm/model_executor/layers/attention/attention.py)
- MosaicML `llm-foundry` 仓库：<https://github.com/mosaicml/llm-foundry>
