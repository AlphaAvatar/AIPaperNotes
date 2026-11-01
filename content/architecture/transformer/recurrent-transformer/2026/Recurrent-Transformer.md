# The Recurrent Transformer: Greater Effective Depth and Efficient Decoding

论文链接：https://arxiv.org/pdf/2604.21215

代码链接：https://github.com/geniucos/recurrent-transformer

## 摘要

Transformer 并行处理 Token，但在时间维度上深度有限：在位置 $t$ 处，每一层关注的是基于上一层计算出的键值（key-value）对，因此其深度受限于层数。循环模型（Recurrent models）虽然具备无限的时间深度，但存在优化不稳定性问题，且以往未能充分利用现代加速器。我们提出了 **Recurrent Transformer**，这是一种简单的架构改进：**每一层关注的是基于自身激活值计算出的键值对**，从而在保持标准自回归解码成本的同时，实现了逐层的循环记忆机制。我们证明，在温和的假设条件下，该架构既能模拟传统 Transformer，又能实现 Token 间的循环更新，同时避免了优化不稳定性。直观上看，预填充（prefill）或训练阶段似乎受限于带宽（有效算术强度接近 1），因为键和值是按顺序生成的；我们提出了一种基于分块（tiling）的精确算法，在保持数学计算逻辑不变的前提下，将 HBM​​ 访存量从 $\Theta(N^2)$ 降低至 $\Theta(N \log N)$，从而将序列长度为 $N$ 时的有效算术强度提升至 $\Theta(N/\log N)$。在基于 C4 数据集的 150M 和 300M 参数规模预训练实验中，循环 Transformer 在交叉熵指标上优于参数量相当的传统 Transformer 基线，且在层数更少（参数总量固定）的情况下实现了这一提升；这表明循环机制允许用深度换取宽度，进而降低了 KV 缓存的内存占用和推理延迟。代码已开源至：https://github.com/geniucos/recurrent-transformer

## 1.Introduction

Transformer 是一类高效的序列模型，但其跨位置的计算在结构上较为浅层：在每一层中，位置 $t$ 关注的是由上一层嵌入计算出的键值对（key-value pairs），这意味着任意两个位置之间在每一层内本质上最多只有一次交互。越来越多的理论研究探讨了注意力模型中有限深度所带来的根本局限性，其中包括利用电路复杂度理论来刻画低深度 Transformer 能够及不能够表达的内容。这些视角促使人们探索能够实现更大有效深度的模型架构。

我们引入了 **Recurrent Transformer**，简称 RT，它通过简单修改键值对（key–value pair）的计算方式，使每一层都具备时间上的循环特性。在标准 Transformer 中，第 $\ell$ 层位置 $t$ 处的键值对是根据该位置在第 $\ell-1$ 层的表征计算得出的，随后可供后续位置 $t' > t$ 进行注意力计算。相比之下，在 Recurrent Transformer 中，第 $\ell$ 层位置 $t$ 处的键值对是根据该位置在第 $\ell$ 层的输出计算得出的，而非基于其在第 $\ell-1$ 层的表征。因此，第 $\ell$ 层的后续位置 $t'$ 在执行注意力计算时，所关注的是位置 $t$ 处已经融合了第 $\ell$ 层注意力机制与 MLP 计算结果的表征。**重要的是，Recurrent Transformer 在每一层内部独立进行这种循环操作，从而使每一层都能维护各自的键值记忆**。这与使用跨层共享记忆的 Feedback Transformer 有所不同；这种层级间的独立性正是我们的架构能够高效实现的关键原因。

我们从表征、优化和计算效率的角度，阐述了 Recurrent Transformer 的设计动机：
1. **Representational perspective**。与标准 Transformer 一样，Recurrent Transformers 保留了针对每个 token 的键值（key–value）记忆；但它通过允许后续位置关注那些已经过注意力机制和 MLP 处理的表征，扩展了单层内可表达的计算空间。在适度假设下，循环 Transformer 既能模拟标准 Transformer 的行为，也能通过将注意力限制在上一位置来实现 token 间的循环计算。这种特性使循环 Transformer 处于完全并行注意力机制与纯粹循环状态空间计算之间，同时避免了固定内存容量带来的瓶颈问题。
2. **Training Stability**。若将模型视为基于位置的有向计算图，传统的 RNN 仅能通过长度为 $(j-i)$ 的中间状态链，将信息从位置 $i$ 传递至位置 $j$。此类路径可能过长，从而引发梯度消失与梯度爆炸问题，导致难以保障远距离位置间的信息流动。Recurrent Transformer 通过引入大量额外的多跳路径（对应于层内各位置间重复进行的“注意力+MLP”操作）缓解了这一问题，同时仍允许任意两个位置之间进行直接的单跳注意力交互。实践表明，该架构若配合键值（key-value）计算前的适当归一化以及标准的逐层残差缩放，能够实现稳定的训练。我们将在第 4 节进一步阐述这一观点，并解释为何该架构通常不会出现梯度爆炸问题。
3. **Training-time efficiency**。针对循环 Transformer（Recurrent Transformer）训练或预填充阶段，一种朴素的实现方式在位置处理上呈顺序性，且往往受限于带宽：键（key）和值（value）是逐位置生成的，而每个 query 都必须对线性增长的前缀部分进行聚合，导致根据 Roofline 模型评估出的有效算术强度极低，仅为 $Θ(1)$。我们提出了一种精确的分块（tiling）算法，该算法在保持原有注意力计算数学逻辑的同时，重组了内存数据移动方式，从而将高带宽内存（HBM）的访存量从 $Θ(N^2)$ 降低至 $Θ(N log N)$，并将有效算术强度提升至 $Θ(N/ log N)$。我们的核心发现是：在训练或预填充过程中，尽管持久化的键值对是按因果顺序（causally）生成的，但完整的 query 序列却是预先已知的。这使得我们能够借鉴 Flash Inference 的思路，将计算过程重组为一种分块调度方案；在该方案中，每个生成的键值块在被移出高速内存之前，会被后续的多个查询重复利用。最终的算法将注意力计算块与 MLP 计算交织进行，并采用 Rabe 和 Staats [2021] 以及 Dao 等人 [2022] 所使用的方法来累积注意力贡献。
4. **Depth to inference efficiency**。至关重要的是，额外的有效时间深度能够带来更优的深度与宽度权衡：在参数总量固定的情况下，若能用较少的层数达到同等质量，便可减少需要存储的键值（key–value）状态量，进而降低解码阶段的内存数据传输量。我们的实验结果印证了这一优势，即层数较浅的循环 Transformer（Recurrent Transformer）模型表现优于层数较深的 Transformer 基线模型。

**贡献**：
- 在第 2 节中，我们提出了 Recurrent Transformer（RT），这是一种逐层循环注意力架构；在该架构中，每一层的键-值对是基于该层的输出（而非上一层的表示）计算得出的。
-  在第 3 节中，我们从表征能力的角度论证了：在温和的假设条件下，Recurrent Transformer 既能模拟标准的自注意力行为，又能通过注意力集中机制实现 Token 到 Token 的循环计算。
- 在第 4 节中，我们对 Recurrent Transformer 的训练稳定性进行了基于路径的分析，展示了该架构如何将额外的多跳计算与直接的单跳注意力路径相结合；同时，我们在简化设定下提供了理论依据，表明在适当的缩放条件下，既不会出现梯度爆炸，也不会出现梯度消失。
- 在第 5 节中，我们针对预填充（prefill）/训练阶段提出了一种精确且感知 I/O 的分块（tiling）算法；该算法在保持数学意义上的注意力计算不变的同时，将内存流量从 $Θ(N^2)$ 降低至 $Θ(N log N)$，并将有效算术强度从 $Θ(1)$ 提升至 $Θ(N/ log N)$。
- 在第 6 节中，我们概述了为使 Recurrent Transformer 的训练更高效、更实用而涉及的各种计算挑战与设计选择。
- 在第 7 节中，我们展示了基于 C4 数据集、参数量为 300M 的模型的实验结果。结果表明，与参数量相当的 Transformer 基线模型相比，该模型在交叉熵指标上表现更优，且在参数量固定的情况下展现出了良好的深度与宽度权衡（如图 2 所示）。具体而言，6 层 Recurrent Transformer 的性能与 12层模型相当（在参数量固定的前提下），同时将 KV 缓存大小减少了约 30%，降低了解码阶段的内存流量，从而提升了推理效率。关于 150M 参数量模型的更多结果，请参阅附录 E.3。

## 2.Architectural overview and notation

**Architectural overview**。与标准的因果（causal）Transformer 相比，Recurrent Transformer 的关键区别在于提供给后续位置的键值对（key–value pairs）的来源。在标准 Transformer 中，位置 $i$ 处的 key-value 对是根据该位置的层输入计算得出的；而在循环 Transformer 中，位置 $i$ 处的持久 key-value 对则是根据该位置的层输出计算得出的。因此，后续位置在进行注意力计算时，所关注的早期位置的表征已历经了同层注意力机制与 MLP（多层感知机）的计算处理，从而使得每一层都能沿时间轴呈现出循环（recurrent）特性。

这在当前位置产生了一种循环依赖：由于位置 $i$ 处的层输出也关注当前位置本身，因此在计算该输出时，无法直接使用持久化键值对 $(k_i, v_i)$。为了解决这一问题，循环 Transformer（Recurrent Transformer）区分了两种键值对。一种是临时键值对，由当前层的输入计算得出，仅在计算当前位置的注意力时使用；另一种是持久化 key-value 对，由计算出的层输出生成，随后被存储起来，供后续所有位置使用。

**Notation**。此处介绍单头（single-head）形式的表述；多头注意力机制则是对每个注意力头独立应用相同的构建方式，随后采用常规的输出投影。设序列长度为 $N$，堆叠层数为 $L$。令 $D$ 为嵌入维度，并考虑输入为 $\textbf x_1, . . . , \textbf x_N ∈ \mathbb R^D$ 的单层结构。令 $MLP : \mathbb R^D → \mathbb R^D$ 表示 MLP 模块，$RMS : \mathbb R^D → \mathbb R^D$ 表示均方根（Root Mean Square）归一化。尽管在实际应用中我们使用可学习参数，但为了便于表述与分析，此处设定 $RMS(x) = \sqrt{D} · x/∥x∥_2$。我们使用（洋红色）$\textcolor{magenta}{RMS}$ 标记来区分针对 Query（查询向量）和 Key（键向量）的归一化操作。

注意力算子 $Attn: (\mathbb R^D × \mathbb R^D)^∗ × \mathbb RD → \mathbb R^D$ 将 key-value 对序列 $(\textbf k_1, \textbf v_1), . . . ,(\textbf k_ℓ, \textbf v_ℓ)$ 和查询 $q$ 映射到

```math
Attn((\textbf k_1, \textbf v_1), . . . ,(\textbf k_ℓ, \textbf v_ℓ), \textbf q)=\sum^{ℓ}_{i=1}\textbf v_i\cdot\frac{exp(⟨\textbf k_i,\textbf q⟩)}{\sum^ℓ_{j=1}exp(⟨\textbf k_j,\textbf q⟩)}
```

我们利用投影矩阵 $Q, K, V \in \mathbb{R}^{D \times D}$，基于嵌入（embedding）计算查询（query）、键（key）和值（value）。遵循标准 Transformer 参数化设定，我们采用 pre 层归一化（pre-norm），并假设注意力机制与 MLP 的残差更新在初始化或参数化时采用了适当的 $1/\sqrt{L}$ 缩放因子，从而确保形式为 $\textbf x \mapsto \textbf x + \frac{1}{\sqrt{L}} \{\text{Attn, MLP}\}(\text{RMS}(\textbf x))$ 的复合映射具有良好的性质。

### 2.1 The Transformer layer

我们首先回顾标准的纯解码器（decoder-only）因果 Transformer 层。给定输入 $\textbf x_1, \dots, \textbf x_N \in \mathbb{R}^D$，位置 $i$ 根据当前层的输入生成其查询（query）、键（key）和值（value）：

```math
\textbf q_i=\textcolor{magenta}{RMS}[Q~RMS(\textbf x_i)],
```

```math
\textbf k_i=\textcolor{magenta}{RMS}[K~RMS(\textbf x_i)],
```

```math
\textbf v_i=V~RMS(\textbf x_i).
```

随后，位置 $i$ 处的注意力输出是通过对该位置之前可用的键值对进行注意力计算而得出的：

```math
\textbf a_i=Attn((\textbf k_1,\textbf v_1),...,(\textbf k_i,\textbf v_i),\textbf q_i).
```

最后，通过将注意力分支与 MLP 残差分支相加，得到该层的输出：

```math
\textbf y_i=\textbf x_i+\frac{1}{\sqrt{L}}\left(\textbf a_i+MLP[RMS(\textbf x_i+\frac{1}{\sqrt{L}}\textbf a_i)]\right).
```

关键的结构特征在于，在标准的 Transformer 中，存储于位置 $i$ 的键值对是根据同一位置的层输入计算得出的。

### 2.2 The Recurrent Transformer layer

<img
  src="https://i-blog.csdnimg.cn/direct/0f7b46813d574078a6e227df994f4a2c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

循环 Transformer 层（如图 1 所示）与标准 Transformer 层的区别，仅在于提供给后续位置的键值对（key–value pairs）的构成方式。在位置 $i$ 处，循环 Transformer 首先利用当前层的输入，结合一个临时键值对来构建查询（query）：

```math
\textbf q_i=\textcolor{magenta}{RMS}[Q~RMS(\textbf x_i)],
```

```math
\textbf k^{\textcolor{red}{temp}}_i=\textcolor{magenta}{RMS}[Q~RMS(\textbf x_i)],
```

```math
\textbf v^{\textcolor{red}{temp}}_i=V~RMS(\textbf x_i).
```

这些定义与 Transformer 在位置 $i$ 处的查询（query）、键（key）和值（value）投影完全相同。随后，利用来自先前位置的持久键值对以及当前位置的临时键值对，计算位置 $i$ 处的注意力输出：

```math
\textbf a_i=Attn((\textbf k_1,\textbf v_1),...,(\textbf k_{i-1},\textbf v_{i-1}),(\textbf k^{\textcolor{red}{temp}}_i,\textbf v^{\textcolor{red}{temp}}_i),\textbf q_i).
```

接下来，我们构建该层的输出表示：

```math
\textbf z_i=\textbf x_i+\frac{1}{\sqrt{L}}\left(\textbf a_i+MLP[RMS(\textbf x_i+\frac{1}{\sqrt{L}}\textbf a_i)]\right),
```

这既是传递给下一层的表示，也是计算位置 $i$ 处持久键值对的源。我们通过对该输出进行投影来定义这一持久键值对：

```math
\textbf k_i=\textcolor{magenta}{RMS}[K~RMS(\textbf z_i)],\tag{1}
```

```math
\textbf v_i=V~RMS(\textbf z_i).\tag{2}
```

因此，$\textbf k^{\textcolor{red}{temp}}_i$ 和 $\textbf v^{\textcolor{red}{temp}}_i$ 仅用于计算位置 $i$ 处的注意力，而不会暴露给后续位置。相比之下，“持久对”（persistent pair）是在 $\textbf z_i$ 形成之后才定义的，并随后被存储以供所有后续位置使用。这意味着，与标准 Transformer 不同，后续位置所关注的并非基于位置 $i$ 处层输入计算得出的“对”，而是基于已更新的表示 $\textbf z_i$ 计算得出的“对”。

我们对临时和持久的键值对复用相同的投影矩阵 $K$ 和 $V$。因此，相比于 Transformer，Recurrent Transformer 并未引入额外的键/值投影参数；这种复用同时也保持了临时与持久键值表示之间的语义共享。

### 2.3 Closest Related Work

在表征层面，与之最相近的是 Feedback Transformer 的变体。Feedback Transformer 采用了一种跨层共享的反馈记忆机制，本质上是基于整个模型的输出（而非各层独立计算）来生成单一的键值对列表。Staircase Attention 则是对 Feedback Transformer 的推广，它探讨了结合权重共享的循环处理与缓存变体——不过，这种处理仍是在模型整体层面而非逐层层面进行的。这种区分不仅在表征层面意义重大，在计算层面也至关重要：在 RT 层内部，所有查询（queries）均可提前获取，这正是我们高效训练方法（见第 5 节）得以实现的关键前提。

TransformerFAM 与之更为相似，因为它同样在每一层独立运行，并允许后续位置访问经过更多处理的表征。然而，它是通过一种受限记忆来实现这一点的，该记忆通过注意力机制进行读写。相比之下，Recurrent Transformer 保留了针对每个 token 的持久化键值（key–value）记忆，而不是将过往信息压缩为固定大小的状态。这种差异至关重要，既有助于避免受限记忆带来的瓶颈，也对第 3.1 节中的表征结果具有重要意义。

## 3. Representational Perspective

本节从理论上证明，在温和的假设条件下，循环 Transformer（Recurrent Transformer）既能模拟 Transformer，也能模拟 RNN。这表明，至少在表征能力方面，循环 Transformer 涵盖了 RNN 和 Transformer。

### 3.1 Representing Transformers

直观地说，循环 Transformer（Recurrent Transformer）可以通过确保由 $\textbf z_i$ 计算得到的持久键值对，与标准 Transformer 通过 $K$ 和 $V$ 投影从 $\textbf x_i$ 计算得到的键值对保持一致，来复现宽度较小的标准 Transformer 的行为。下面我们将这一表述具体化：

**定理 1**（非正式）。*任意宽度为 $d'$ 的 Transformer，都可以由宽度为 $d = 3d'$ 的循环 Transformer 近似模拟：被模拟的 Transformer 的激活可以嵌入到 RT 嵌入表示中互不重叠的特征组内。可以对 RT 层进行参数化，使得（i）注意力分数保持不变，且（ii）该层的输出精确地跟随 Transformer 层的输出。*

从总体上看，这一构造通过将互不重叠的特征块分配给不同角色，在 RT 更大的嵌入表示中表示较小 Transformer 的状态。其中一个特征块存储层输入 $\textbf x_i$ 的受保护副本，使得 RT 随后从层输出 $\textbf z_i$ 计算持久键和值时，仍能恢复出与 Transformer 从 $\textbf x_i$ 计算得到的完全相同的键值对。另一个独立的特征块用于存放注意力贡献 $\textbf a_i$，这样，在应用 MLP 之前将其加到 $\textbf x_i$ 上时，$\textbf x_i$ 的内容就能得到保护而不会丢失。由此，后续 token 所得到的注意力分数完全相同，而指定特征块中的层输出也与 Transformer 的层输出一致。宽度开销为 3 倍而非 2 倍，是堆叠多个层时一项微妙的技术要求；单层则可以用 2 倍的宽度开销来复现。完整构造和形式化证明见附录 A.1。
## 3.2 Representing token-to-token recurrence

如果通过位置嵌入或偏置，使注意力局部集中于前一个位置，那么循环 Transformer 就会实现类似 RNN 的更新。形式化地说，如果 $\mathbf{a}_i$ 主要由前一个持久值 $\mathbf{v}_{i-1}$ 决定，即 $\langle \mathbf{q}_i,\mathbf{k}_{i-1}\rangle \gg \langle \mathbf{q}_i,\mathbf{k}_i^{\mathrm{temp}}\rangle$，且对于任意 $j < i-1$，都有 $\langle \mathbf{q}_i,\mathbf{k}_{i-1}\rangle \gg \langle \mathbf{q}_i,\mathbf{k}_j\rangle$，则有：

```math
\mathbf{z}_i \approx \mathbf{x}_i + \mathbf{v}_{i-1} + \operatorname{MLP}[\operatorname{RMS}(\mathbf{x}_i + \mathbf{v}_{i-1})]
= V\operatorname{RMS}(\mathbf{z}_{i-1}) + \mathbf{x}_i + \operatorname{MLP}[\operatorname{RMS}(\mathbf{x}_i + V\operatorname{RMS}(\mathbf{z}_{i-1}))]
```

在进一步作出简化假设，即 $V$ 为恒等映射的情况下，上式变为一种带有跳跃连接的特定状态递推：

```math
\mathbf{z}_i = \operatorname{RMS}(\mathbf{z}_{i-1}) + \mathbf{x}_i + \operatorname{MLP}[\operatorname{RMS}(\mathbf{x}_i + \operatorname{RMS}(\mathbf{z}_{i-1}))]
```

我们并不声称能够复现门控 RNN/LSTM，也不声称训练会使模型学到此类结构。我们强调的是，从表示能力的角度来看，循环 Transformer 足够丰富，能够在单个层内表达显式的迭代计算，同时保留整个前缀中每个 token 的记忆（这是上一节模拟 Transformer 所必需的）。

关键在于，一旦一种架构能够表示这样的迭代计算，一个自然的问题便是：**RNN 的经典可学习性问题是否会阻碍训练**。第 4 节解释了为什么循环 Transformer 的多跳动态仍然能够稳定训练。

### 3.3 Why temporal depth matters

Transformer 在时间维度上的深度较浅：沿序列进行的更深层迭代计算，必须主要通过堆叠网络层来模拟。关于低深度注意力模型和有限自动机状态跟踪问题的理论表明，深度受限会产生具体影响：浅层 Transformer 的表示能力不足以模拟某些自动机；更一般地，其能力受限于 $\mathrm{TC}^{0}$——一类浅层电路 [Merrill et al., 2022]。循环 Transformer 在每一层内部引入了额外的时间深度。这与堆叠网络层所获得的深度形成互补，因此有望在使用更少网络层的同时，达到与 Transformer 相当的有效深度。我们在第 7 节通过实验验证了这一假设。

## 4.Training Stability of Recurrent Transformer

本节解释循环 Transformer 如何避免梯度随深度增加而消失或爆炸等退化动态。我们将模型视为以序列位置为节点的有向计算图，从而形式化我们的论述：当位置 $j$ 的计算*直接*依赖于位置 $i$ 计算得到的量时，就存在一条边 $i \to j$。在经典 RNN 中，从位置 $i$ 到 $j$ 的信息（以及梯度）必须经过完整的链 $i \to i+1 \to \cdots \to j$，而沿长链反复复合会导致梯度消失或爆炸现象。这种链式拓扑迫使从位置 $i$ 到 $j$ 的所有影响都经过 $(j-i)$ 次连续的状态转移。要使训练稳定，通常需要这些转移接近收缩映射，但这又会使 $\mathbf{x}_i$ 对 $\mathbf{x}_j$ 的影响随 $(j-i)$ 的增大而迅速减弱，导致远距离信息难以传递。虽然在 RNN 中可以通过精心设计的初始化方案缓解这一问题，但我们的方法利用了这样一个事实：既存在直接跳转，也存在额外的跨层多跳路径。

与标准 Transformer 一样，token $j$ 可以通过存储的键值对 $(\mathbf{k}_i,\mathbf{v}_i)$，直接关注任意更早的 token $i < j$，从而形成一条单跳信息路径 $i \to j$。关键区别在于，在循环 Transformer 中，存储的键值对 $(\mathbf{k}_i,\mathbf{v}_i)$ 是由*层输出* $\mathbf{z}_i$ 计算得到的，而 $\mathbf{z}_i$ 已经包含了对更早存储的键值对进行注意力计算的结果。因此，信息不仅可以从 $i$ 直接传播到 $j$，还可以*间接传播*。

具体来说，从 token 1 到 token 4 的多跳路径（图 1）可以经过中间的写入—读取步骤：

```math
\mathbf{x}_1 \to \mathbf{z}_1 \to (\mathbf{k}_1,\mathbf{v}_1) \to \mathbf{a}_2 \to \mathbf{z}_2 \to (\mathbf{k}_2,\mathbf{v}_2) \to \mathbf{a}_4 \to \mathbf{z}_4
```

这里，每一步 $\mathbf{z}_t \to (\mathbf{k}_t,\mathbf{v}_t)$ 都是向该层的持久记忆进行一次*写入*，而每一步 $(\mathbf{k}_t,\mathbf{v}_t) \to \mathbf{a}_{t'}$ 都是在任意后续位置 $t' > t$ 通过注意力进行一次*读取*。将这些写入—读取操作串联起来，就会产生长度随位置间距离增长的多跳影响路径，从而实现层内迭代计算（如第 3.2 节所述），同时保留 Transformer 的直接单跳注意力路径。

**衰减长路径，同时保留长距离访问。** 多跳路径只有在不发生爆炸时才有用。在实践中，两个调节因素起主导作用。首先，对残差分支采用标准的随深度缩放方式，使每层的更新保持在稳定范围内 [Bordelon et al., 2023, Yang et al., 2023]。其次，在计算持久键和值之前进行归一化（公式 1–2 中的 $\operatorname{RMS}(\mathbf{z}_i)$），即使 $\mathbf{z}_i$ 是多个分量之和，也能控制其幅度。实验表明，这些设计使长多跳路径的影响趋于衰减：链越长，影响越小。与纯 RNN 不同，这并不会消除长距离访问能力，因为即使长链受到衰减，直接注意力边仍然存在。

在下面的定理中，针对一个不含归一化的高度简化设置，我们证明初始化时不会出现梯度爆炸。归一化有助于提高稳定性，也就是说，它允许循环 Transformer 在更高的学习率下稳定训练。附录 E.2 通过实验展示了这一点。

**定理 4.1。** *考虑一个简化的单层 RT 层，它仅包含均匀注意力，输入为 $x_1,\ldots,x_n$，输出记为 $z_1,\ldots,z_n$，其中*

```math
z_k = x_k + \frac{\alpha}{k}\left(Vx_k + V\sum_{j=1}^{k-1}z_j\right)
```

*其中，$\alpha$ 是表示残差缩放的标量，$V$ 是值矩阵。那么，对于 $k \geq 2$，有*

```math
\frac{\partial z_k}{\partial x_1}
= \frac{1}{k!}\sum_{r=1}^{k}
\begin{bmatrix}k\\r\end{bmatrix}
\alpha^r V^r
```

*其中，$\begin{bmatrix}k\\r\end{bmatrix}$ 表示 $k$ 个元素的所有排列中，恰好具有 $r$ 个循环的排列总数。*

由于排列总数为 $k!$，上述定理表明，只要 $\alpha V$ 的最大特征值小于 1，从 $z_j$ 到 $x_1$ 就不会出现梯度爆炸。因此，对于正交归一初始化，只要 $\alpha < 1$，我们就预期处于这一情形。此外，由于总梯度是对不同长度路径的贡献求和（路径长度由上式中的 $r$ 表示），可以看出，即使 $\alpha V$ 的最大特征值小于 1，梯度也不会消失。特别地，由于 $\begin{bmatrix}k\\1\end{bmatrix} = (k-1)!$，上式中对应于 $r=1$ 的项为 $\frac{\alpha}{k}\cdot V$，这恰好是普通 Transformer 所产生的梯度。该定理的证明见附录 B。

## 5. Exact Tiling for Training and Prefill

## 6. A deep dive into the computational challenges

### 6.1 MLPs and batch size

### 6.2 Using CUDA Graphs

### 6.3 Cache locality and memory access pattern

### 6.4 The backward pass
