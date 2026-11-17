# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

论文链接：https://arxiv.org/pdf/2505.06708

代码链接：

## 摘要

门控机制已被广泛应用，从 LSTM 和 Highway Networks 等早期模型，到近期的状态空间模型、线性注意力机制以及 Softmax 注意力机制，皆可见其身影。然而，现有文献鲜有探讨门控机制的具体作用。在本研究中，我们开展了全面实验，系统地考察了引入门控机制的 Softmax 注意力变体。具体而言，我们在一个包含 3.5T Token 的数据集上，对 15B 参数规模的混合专家模型（MoE）和 1.7B 参数规模的稠密（dense）模型中的 30 种变体进行了详尽对比。我们的核心发现是，一项简单的改进——**即在“缩放点积注意力”（Scaled Dot-Product Attention, SDPA）之后应用一个“头（head）”特定的 Sigmoid 门控**——能够持续提升模型性能。该改进还能增强训练稳定性，支持更大的学习率，并改善模型的扩展（scaling）特性。通过对比不同的门控位置及计算变体，我们将这种有效性归因于两个关键因素：(1) 在 Softmax 注意力的低秩映射中引入了非线性；(2) 采用依赖于 query 的稀疏门控分数来调节 SDPA 的输出。值得注意的是，我们发现这种稀疏门控机制能够缓解 attention sink 现象，并提升长上下文的外推（extrapolation）能力。此外，我们还开源了相关代码和模型，以促进后续研究。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/5acfea429b674a97a0140da31d1c2371.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

门控机制在神经网络中已得到广泛应用。早期的网络架构（如 LSTM、Highway Network 和 GRU）率先引入了门控机制，旨在控制跨时间步骤或跨层的信息流动，并改善梯度传播。这一原则在现代网络架构中依然沿用。近期的一些序列建模方法（包括状态空间模型和注意力机制）也普遍采用了门控机制，通常用于调节“Token 混合”（token-mixer）组件的输出。尽管门控机制已被广泛采用并取得了良好的实证效果，但除了最初提出的直观构想外，人们对其具体功能和影响的深入研究仍显不足。

对“门控”（gating）机制理解不足，阻碍了对其真实贡献的评估，尤其是在它与其他架构因素相互交织的情况下。例如，尽管 Switch Heads 引入了 sigmoid 门控来筛选 Top-K 注意力头专家（attention head experts），但我们的实验却揭示了一个有趣的现象（见附录 A.1）：即便将专家数量缩减至仅剩一个——此时门控仅起到调节数值输出的作用——模型性能依然能获得显著提升。这有力地表明，门控机制本身就具有重要的内在价值，且这种价值独立于路由机制之外。同样，在“原生稀疏注意力”（Native Sparse Attention, NSA）中，尽管模型整体性能有所提升，但研究并未将门控机制的贡献与稀疏注意力设计本身所带来的影响区分开来。这些考量凸显了**严格剥离门控机制与其他架构组件各自影响的必要性**。

在本研究中，我们探讨了标准 Softmax 注意力机制中的门控机制（第 2.2 节）。具体而言，我们在不同位置引入了门控（图 1）：在 Query ($G_4$)、Key ($G_3$) 和 Value ($G_2$) 的投影之后；在缩放点积注意力 (SDPA) 的输出之后 ($G_1$)；以及在最终的稠密输出层之后 ($G_5$)。我们考察了多种门控变体，包括逐元素 (elementwise) 与逐头 (headwise) 门控、头专用 (head-specific) 与头共享 (head-shared) 门控，以及加法型与乘法型门控。研究发现：(i) 在 SDPA 输出端应用“头专用”门控 ($G_1$) 可带来最显著的性能提升（例如，困惑度 PPL 最多可降低 0.2，MMLU 得分可提升 2 个点）；(ii) SDPA 输出端的门控还能改善训练稳定性，几乎消除了损失值激增 (loss spikes) 的现象，从而支持使用更大的学习率，并增强了模型的可扩展性。

我们确定了影响门控（gating）有效性的两个主要因素：(i) **非线性**。两个连续的线性层——即值（value）投影（$W_v$）和稠密（dense）投影（$W_O$）——可以合并为一个低秩线性投影。因此，在 $G_1$ 或 $G_2$ 位置通过门控引入非线性，能够增强这种低秩线性变换的表达能力（第 4.1 节）。(ii) **稀疏性**。尽管非线性门控变体始终能提升性能，但我们观察到其增益程度各不相同。进一步分析表明，门控得分所呈现出的显著稀疏性是另一个关键因素，它为缩放点积注意力（SDPA）的输出引入了依赖于输入的稀疏性（第 4.2 节）。此外，稀疏门控还消除了“注意力汇聚”（attention sink）现象，即初始标记（initial tokens）在注意力得分中占据了不成比例的主导地位（图 2，第 4.3 节）。既往研究将“注意力汇聚”解释为由非负 Softmax 归一化导致的冗余注意力累积。通过实证研究，我们验证了当在 SDPA 输出端应用依赖于查询（query）的稀疏门控时，无论是稠密模型还是混合专家（MoE）模型（均在 3.5T 标记数据上训练），均不再出现“注意力汇聚”现象。不仅如此，这些模型在长度泛化方面表现优异，在 RULER 基准测试中取得了超过 10 个点的性能提升（第 4.4 节）。

总之，我们的研究凸显了在标准注意力层中引入门控机制对模型性能及行为的影响。通过评估不同的门控变体，我们发现它们能够引入非线性和稀疏性，并消除“注意力汇聚点”（attention sinks）。这些发现加深了我们对门控注意力机制的理解。我们将开源这些不含“注意力汇聚点”的模型，以推动未来的相关研究。

## 2.Gated-Attention Layer

### 2.1 Preliminary: Multi-Head Softmax Attention

给定输入 $X \in \mathbb{R}^{n \times d_{\text{model}}}$（其中 $n$ 为序列长度，$d_{\text{model}}$ 为模型维度），Transformer 注意力层的计算可分为四个阶段。

**QKV Linear Projections**。输入 $X$ 通过学习得到的权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$ 进行线性变换，得到查询向量 Q、键向量 K 和值向量 V，其中 $Q, K, V \in \mathbb{R}^{n \times d_k}$：

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.\tag{1}
```

**Scaled Product Dot-Product Attention (SDPA)**。计算查询（queries）与键（keys）之间的注意力得分，随后进行 softmax 归一化。输出即为各值（values）的加权和：

```math
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V,\tag{2}
```

其中 $\frac{QK^T}{\sqrt{d_k}}\in \mathbb{R}^{n \times n}$ 表示缩放点积相似度矩阵，而 $\text{softmax}(\cdot)$ 则确保注意力权重均为非负数，且每一行的权重之和为 1。

**Multi-Head Concatenation**。在多头注意力机制中，上述过程针对 $h$ 个“头”（heads）并行重复执行；每个“头”都拥有各自的投影矩阵 $W_q^i$、$W_k^i$ 和 $W_v^i$。所有“头”的输出会被拼接在一起：

```math
MultiHead(Q,K,V)=Concat(head_1,...,head_h),\tag{3}
```

其中 $head_i=Attention(QW^i_Q,KW^i_K,VW^i_V)$。

**Final Output Layer**。级联后的 SDPA 输出通过输出层 $W_o ∈ \mathbb R^{hd_k×d_{model}}$ 传递：

```math
O=MultiHead(Q,K,V)W_o.\tag{4}
```

### 2.2 Augmenting Attention Layer with Gating Mechanisms

门控机制形式化为：

```math
Y'=g(Y,X,W_{\theta},\sigma)=Y\odot\sigma(XW_{\theta}),\tag{5}
```

其中，$Y$ 是待调制输入，$X$ 是用于计算门控得分的另一个输入，$W_θ$ 代表门控机制中的可学习参数，$σ$ 为激活函数（例如 sigmoid 函数），而 $Y′$ 则是经过门控处理后的输出。门控得分 $σ(XW_θ)$ 实际上起到了动态滤波器的作用，通过选择性地保留或抹除 $Y$ 中的特征，来控制来自 $Y$ 的信息流。

在本研究中，我们全面探讨了注意力层中门控机制的几种变体。我们的研究重点在于以下五个关键方面：(1) **位置**。我们研究了在不同位置应用门控机制的效果，如图 1（左）所示：(a) 在 Q、K、V 投影之后（公式 1），对应于图 1（左）中的 $G_2$、$G_3$ 和 $G_4$ 位置；(b) 在缩放点积注意力（SDPA，公式 3）的输出之后（$G_1$）；$\text{(c)}$ 在最终拼接后的多头注意力输出之后（公式 4，$G_5$）。(2) **粒度**。我们考虑了门控分数的两种粒度：(a) 头粒度（Headwise）：使用单个标量门控分数来调节整个注意力头的输出。(b) 元素粒度（Elementwise）：门控分数是与 $Y$ 维度相同的向量，能够实现细粒度的、逐维度的调节。(3) **头独立或共享**。鉴于注意力机制的多头特性，我们进一步考虑：(a) 头独立（Head-Specific）：每个注意力头拥有各自的门控分数，从而实现各注意力头的独立调节。(b) 头共享（Head-Shared）：$W_θ$ 和门控分数在所有注意力头之间共享。(4) **乘法或加法**。在将门控分数应用于 $Y$ 时，我们考虑：(a) 乘法门控：门控后的输出 $Y′$ 计算方式为 $Y′ = Y · σ(Xθ)$。(b) 加法门控：$Y′ = Y + σ(Xθ)$。(5) **激活函数**。我们主要考虑两种常见的激活函数：SiLU 和 sigmoid。对于加法门控，我们仅使用 SiLU（因为其输出范围无界），而 sigmoid 的输出值仅限于 [0, 1] 区间。此外，为了进一步剖析门控机制发挥作用的内在原理，我们还考察了恒等映射（Identity Mapping）或 RMSNorm（详见第 4.1 节）。

除非另有说明，我们采用 head-specific 的乘法门控机制，并使用 Sigmoid 激活函数（$σ(x) = \frac{1}{(1+e)}$）。

## 3.Experiments
