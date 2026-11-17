# ATTENTION RESIDUALS

论文链接：https://arxiv.org/pdf/2603.15031

代码链接：https://github.com/MoonshotAI/Attention-Residuals

## 摘要

在现代大语言模型（LLM）中，采用 PreNorm 的残差连接已成为标准配置；然而，这种机制仅通过固定的单位权重将各层的输出进行累加。这种统一的聚合方式会导致隐藏状态随网络深度增加而失控增长，并逐渐稀释各层原本的贡献。为此，我们提出了 **Attention Residuals**（*AttnRes*）机制：它不再采用固定的累加方式，而是对先前各层的输出执行 Softmax 注意力操作，使每一层都能根据输入内容，通过学习到的权重有选择地聚合之前的表征。针对大规模模型训练中因关注所有先前层输出而带来的内存与通信开销问题，我们进一步引入了“**分块注意力残差**”（Block AttnRes）——将网络层划分为若干区块，并仅对区块级别的表征进行注意力计算。该方法在保留完整 AttnRes 大部分性能增益的同时，有效降低了内存占用。结合基于缓存的流水线通信和两阶段计算策略，Block AttnRes 可作为标准残差连接的实用替代方案，且仅引入极小的额外开销。缩放定律（Scaling Law）实验表明，该改进在不同模型规模下均表现稳健；消融实验则验证了基于内容进行深度维度选择的有效性。此外，我们将 AttnRes 集成至 Kimi Linear 架构（总参数量 48B，激活参数量 3B）中，并在 1.4T Token 的数据上进行了预训练。结果显示，AttnRes 有效缓解了 PreNorm 带来的稀释效应，使得各层输出幅值与梯度分布更加均匀，并提升了模型在各项下游任务中的表现。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/d5b6a9300c754e749761bcb0ebcda26f.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

标准的残差连接是现代大语言模型（LLM）中公认的基础构建模块。更新公式 $\textbf h_l = \textbf h_{l-1} + f_{l-1}(\textbf h_{l-1})$ 通常被视为一条“梯度高速公路”：通过**恒等映射**，梯度能够绕过中间的变换过程，从而确保深层网络在训练时的稳定性。然而，残差连接还发挥着另一个较少受到关注的作用。**将这种递归关系展开后可以发现，每一层接收到的输入，实际上都是之前所有层输出的等权重总和**；换言之，残差连接决定了信息如何在网络深度方向上进行聚合。与序列混合（sequence mixing）或专家路由（expert routing）——这些机制现已采用基于输入且可学习的加权方式——不同，这种深度方向上的信息聚合仍由固定的单位权重所决定，缺乏一种能够选择性地强化或抑制特定层贡献的机制。

在实践中，PreNorm 已成为主流范式；然而，由于其采用无权累加方式，隐藏状态的幅值会随深度以 $O(L)$ 的速度增长，导致各层对最终输出的相对贡献被逐渐稀释。早期层的信息因此被掩盖，且无法被选择性地提取；**从经验上看，在性能损失极小的情况下，模型中相当大比例的层是可以被剪枝的**。近期的一些尝试（如缩放残差路径和多流循环结构）本质上仍未脱离“加法式循环”的框架，而那些引入了跨层访问机制的方法，又往往难以扩展。这种情况与循环神经网络（RNN）在引入注意力机制之前所面临的挑战如出一辙——当时，RNN 在处理序列维度时也遇到了类似的困境。

我们观察到，在深度方向上的累积与循环神经网络（RNN）中的序列递归之间存在一种形式上的对偶性。基于这一对偶性，我们提出了“注意力残差”（**Attention Residuals**，简称 *AttnRes*）。该机制将原本固定的累积方式 $\textbf h_l = \sum_i \textbf v_i$ 替换为 $\textbf h_l = \sum_i \textcolor{red}{\alpha_{i \to l}}\cdot \textbf v_i$，其中 $\textcolor{red}{\alpha_{i \to l}}$ 是通过每层一个可学习的“伪查询”（pseudo-query）向量 $\textbf w_l \in \mathbb{R}^d$ 计算得出的 softmax 注意力权重。这种轻量级机制仅需每层一个 $d$ 维向量，即可实现跨深度的、基于内容感知的选择性检索。事实上，标准的残差连接以及先前基于递归的变体，本质上都可以被视为在深度维度上执行了“线性注意力”；而 AttnRes 则将其推广为“深度方向的 softmax 注意力”。这一转变在序列建模中已被证明具有变革性意义，而 AttnRes 将同样的“从线性到 softmax”的演进引入了深度维度（见第 6.2 节、第 6.1 节）。

在标准训练中，Full AttnRes 带来的额外开销微乎其微，因为其所需的层输出本就已在反向传播过程中被保留。然而，在大规模训练场景下，通常会采用激活值重计算（activation recomputation）和流水线并行（pipeline parallelism）技术；在这种情况下，这些激活值必须被显式地保存，并在流水线各阶段间进行传输。为了在此类场景下保持高效，我们提出了 Block AttnRes：将各层划分为 $N$ 个块（block），通过标准残差连接将每个块内的信息归约为单一表征，并仅针对这 $N$ 个块级摘要（block-level summaries）执行跨块注意力机制。该方法将内存占用和通信开销均降至 $O(Nd)$；结合基础设施层面的优化（见第 4 节），Block AttnRes 可直接替代标准的残差连接，且仅会增加极少的训练成本，推理延迟开销也几乎可以忽略不计。

缩放定律（Scaling Law）实验证实，在各类计算预算下，AttnRes 的表现始终优于基线模型；其中，Block AttnRes 所达到的损失值（loss），与使用 1.25 倍计算资源训练的基线模型相当。我们进一步将 AttnRes 整合至 Kimi Linear 架构（总参数量 48B，激活参数量 3B）中，并在 1.4T token 的数据上进行了预训练。对训练动态的分析表明，AttnRes 有效缓解了 PreNorm 带来的“稀释”效应：模型输出的量级在不同深度下均保持在有限范围内，且梯度范数在各层间的分布也更加均匀。在下游基准测试中，我们的最终模型在所有评估任务上的表现均优于基线模型。

**Contributions**：
- **Attention Residuals**。我们提出了 AttnRes，它用基于学习的“深度维度 Softmax 注意力”取代了固定的残差累加机制；此外，我们还提出了其可扩展变体 Block AttnRes，将内存和通信开销从 $O(Ld)$ 降低至 $O(Nd)$。通过统一的结构化矩阵分析，我们证明了标准的残差结构以及先前基于循环（recurrence）的变体，本质上对应于“深度维度线性注意力”，而 AttnRes 则实现了“深度维度 Softmax 注意力”。
- **Infrastructure for scale**。我们开发了多项系统优化技术，使得 Block AttnRes 在大规模应用中既实用又高效。这些优化包括：在流水线并行（pipeline parallelism）下消除冗余数据传输的“跨阶段缓存”机制，以及通过在线 Softmax（online softmax）来分摊跨块注意力（cross-block attention）计算开销的“两阶段推理”策略。采用这些优化后，训练开销微乎其微，而在典型的推理负载下，推理延迟的增加幅度不到 2%。
- **Comprehensive evaluation and analysis**。我们通过缩放定律（scaling law）实验、组件消融研究以及下游基准测试，对 AttnRes 进行了验证。实验基于一个在 1.4T token 上预训练的 48B 参数模型，结果表明，与标准残差连接相比，AttnRes 实现了持续的性能提升。训练动态分析进一步揭示，AttnRes 能够缓解 PreNorm 带来的特征稀释问题，使隐藏状态的幅值保持在有限范围内，并促使梯度在网络深度方向上的分布更加均匀。

## 2.Motivation

**Notation**。考虑一批形状为 $B \times T \times d$ 的输入序列，其中 $B$ 为批次大小（batch size），$T$ 为序列长度，而 $d$ 为隐藏层维度。为清晰起见，我们针对单个 token 给出公式：$\textbf h_l \in \mathbb{R}^d$ 表示进入第 $l$ 层的隐藏状态，其中 $l \in \{1, \dots, L\}$ 为层索引，$L$ 为总层数。token 嵌入即为 $\textbf h_1$。函数 $f_l$ 表示第 $l$ 层所执行的变换。在 Transformer 模型中，我们将每一个自注意力（self-attention）模块或多层感知机（MLP）均视为独立的一层。

### 2.1 Training Deep Networks via Residuals

**Residual Learning**。残差学习被证明是训练深度网络的一项关键技术，因为它允许梯度绕过变换过程。具体而言，每一层对隐藏状态的更新方式如下：

```math
\textbf h_l=\textbf h_{l-1}+f_{l-1}(\textbf h_{l-1})
```

展开这一递归关系可知，第 $l$ 层的隐藏状态是嵌入（embedding）与之前所有层输出之和：$\textbf h_l = \textbf h_1 + \sum_{i=1}^{l-1} f_i(\textbf h_i)$。残差连接（residual connections）背后的核心思想是“恒等映射”（identity mapping）：每一层都保留了一条直接通路，使得信息和梯度能够无损地传递。在反向传播过程中，关于中间隐藏状态的梯度为：

```math
\frac{∂\mathcal L}{∂\textbf h_l}=\frac{∂\mathcal L}{∂\textbf h_L}\cdot\prod^{L-1}_{j=l}\left(\textbf I+\frac{∂f_j}{∂\textbf h_j}\right)
```

展开该乘积可得到单位项 $\textbf I$ 以及包含各层雅可比矩阵 $\partial f_j / \partial \textbf h_j$ 的高阶项。单位项始终得以保留，从而提供了一条从损失函数到任意层（无论深度如何）的直接梯度路径。

**Generalizing Residual**。尽管残差更新中采用固定的单元系数行之有效，但这种做法将每一层的贡献同等对待，缺乏一种能够根据网络深度调整融合方式的机制。高速网络（Highway networks）通过引入可学习的逐元素门控机制，放宽了这一限制：

```math
\textbf h_l=(1-\textbf g_l)\odot\textbf h_{l-1}+\textbf g_l\odot f_{l-1}(\textbf h_{l-1})
```

其中 $\textbf g_l \in [0, 1]^d$ 在变换路径与恒等路径之间进行插值。更一般地，两者均可视为加权递归形式 $\textbf h_l = \alpha_l \cdot \textbf h_{l-1} + \beta_l \cdot f_{l-1}(\textbf h_{l-1})$ 的特例：当 $\alpha_l = \beta_l = 1$ 时，对应残差（residual）设定；当 $\alpha_l = 1 - \textbf g_l$ 且 $\beta_l = \textbf g_l$ 时，则对应高速公路（Highway）设定。

**Limitations**。无论是固定式还是门控式，这两种方法都存在一个根本性的局限：每一层只能访问其直接输入 $\textbf h_{l-1}$——这是一个将之前所有层的输出汇总而成的单一压缩状态，而非各层原本独立的输出。这导致了以下几个问题：(1) 无法进行选择性访问：不同类型的层（例如注意力层与多层感知机 MLP）接收到的是相同的汇总状态，尽管它们本可以从不同的加权方式中获益；(2) 不可逆的信息丢失：在汇总过程中丢失的信息，无法在更深层中被选择性地恢复；(3) 输出规模增长：后续层为了对累积残差施加更大影响，往往会学习出规模越来越大的输出，这可能会导致训练过程变得不稳定。这些局限性促使我们寻求一种机制，使每一层都能从之前所有的层中选择性地汇总信息。

## 3.Attention Residuals: A Unified View of Time and Depth

上述局限性令人联想到序列建模中存在的类似瓶颈，这提示我们可以在深度维度上寻求类似的解决方案。

**The Duality of Time and Depth**。与循环神经网络（RNN）在时间维度上的运作方式类似，残差连接将之前所有的信息压缩至深度维度上的单一状态 $\textbf h_l$ 中。在序列建模领域，Transformer 通过引入“注意力机制”取代了循环结构，从而实现了对 RNN 的改进；这使得序列中的每个位置都能根据数据相关的权重，有选择地访问之前所有的位置。我们针对深度维度提出了同样的思路：

```math
\textbf h_l=\textcolor{red}{\alpha_{0\to l}}\textbf h_1+\sum^{l-1}_{i=1}\textcolor{red}{\alpha_{i\to l}}\cdot f_i(\textbf h_i)\tag{1}
```

其中 $\textcolor{red}{\alpha_{i \to l}}$ 为各层特定的注意力权重，满足 $\sum_{i=0}^{l-1}\textcolor{red}{\alpha_{i \to l}} = 1$。与序列长度（可达数百万个 token）不同，网络深度通常较小（$L < 1000$），这使得在深度维度上进行 $O(L^2)$ 复杂度的注意力计算在算力上是可行的。我们将这种方法称为 **Attention Residuals**，简写为 *AttnRes*。

### 3.1  Full Attention Residuals

注意力权重可表示为 $\textcolor{red}{\alpha_{i \to l}} = \phi(\textbf q_l, \textbf k_i)$，其中 $\phi: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ 为核函数，$\textbf q_l$ 和 $\textbf k_i$ 分别是查询向量（query vector）和键向量（key vector）。选择不同的 $\phi$ 函数，可以推导出不同的残差变体（见第 6.2 节）；我们采用 $\phi(\textbf q, \textbf k) = \exp(\textbf q^\top \text{RMSNorm}(\textbf k))$（包含归一化操作），从而在深度维度上实现了 softmax 注意力机制：

```math
\textcolor{red}{\alpha_{i\to l}}=\frac{\phi(\textbf q_l,\textbf k_i)}{\sum^{l-1}_{j=0}\phi(\textbf q_l,\textbf k_j)}\tag{2}
```

对于每一层 $l$，我们定义：

```math
\textbf q_l=\textbf w_l,\quad\textbf k_i=\textbf v_i=\begin{cases}
\textbf h_1 & i=0\\
f_i(\textbf h_i) & 1\le i\le l-1
\end{cases}\tag{3}
```

其中查询向量 $\textbf q_l = \textbf w_l$ 是 $\mathbb{R}^d$ 空间中与层相关的可学习向量。$\phi$ 内部的 RMSNorm 机制可防止输出幅值较大的层主导注意力权重。因此，第 $l$ 层的输入为：

```math
\textbf h_l=\sum^{l-1}_{i=0}\textcolor{red}{\alpha_{i\to l}}\cdot\textbf v_i\tag{4}
```

我们将这种形式称为“全注意力残差”（Full Attention Residuals）。对于每个 token，“全注意力残差”需要 $O(L^2d)$ 的计算量，并占用 $O(Ld)$ 的内存来存储各层的输出。由于网络深度远小于序列长度，因此其计算开销相对较小。

**Overhead**。$O(Ld)$ 内存占用与反向传播过程中已保留的激活值完全重叠，因此在常规训练中，Full AttnRes 不会引入额外的内存开销。然而，在大规模训练场景下，通常会采用“激活值重计算”和“流水线并行”技术：原本会被释放并可在后续重新计算的层输出，现在必须被保留以供后续所有层使用；而在流水线并行模式下，这些数据还必须在不同的流水线阶段之间进行传输。在这种情况下，内存开销和通信开销均会随之增长，达到 $O(Ld)$ 的量级。

**Blockwise optimization**。在 Full AttnRes 中，一项经过深思熟虑的设计选择是：将伪查询（pseudo-query）$\textbf w_l$ 设为一个可学习参数，使其与该层的正向计算解耦。这种独立性意味着，任意一组层的注意力权重均可并行计算，无需等待其各自的顺序输出；具体而言，它允许将 $L$ 个层划分为 $N$ 个块（每个块包含 $S$ 个层），并在每个块内对注意力计算进行批处理。这使得单层内存 I/O 开销从 $O(Ld)$ 降低至 $O((S+N)d)$（关于该两阶段策略的详细说明，请参阅第 4 节）。然而，在当前的分布式训练框架下，主要瓶颈并非本地内存带宽，而是流水线并行（pipeline parallelism）中的跨阶段通信——每一层的输出仍需在不同阶段间传输，这种 $O(Ld)$ 的通信开销无法通过本地批处理来缓解。基于此，下文引入了 Block AttnRes 变体，旨在将跨阶段表示的数量从 $L$ 减少至 $N$。我们预期，随着未来互连技术的改进，完整的 $O(Ld)$ 通信将变得切实可行，从而充分释放 Full AttnRes 的潜力。

### 3.2 Block Attention Residuals

<img
  src="https://i-blog.csdnimg.cn/direct/fd16a4625ea14692a83e6a695c2752f3.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们提出了“块注意力残差”（Block Attention Residuals）机制，将 $L$ 个层划分为 $N$ 个块：在每个块内，各层的输出通过求和被归约为单一表征；而在块与块之间，我们仅针对这 $N$ 个块级表征以及 token 嵌入（token embedding）执行全注意力计算。该方法将内存开销和通信开销均从 $O(Ld)$ 降低至 $O(Nd)$。

**Intra-Block Accumulation**。具体而言，我们将 $L$ 层划分为 $N$ 个区块，每个区块包含 $S = L/N$ 层（假设 $L$ 能被 $N$ 整除；若不能整除，则最后一个区块包含剩余的 $L~mod~N$ 层）。令 $\mathcal B_n$ 表示第 $n$ 个区块（$n = 1, . . . , N$）中各层的索引集合。为了构成一个区块，我们将该区块内所有层的输出进行求和：

```math
\textbf b_n=\sum_{j\in\mathcal B_n}f_j(\textbf h_j)\tag{5}
```

我们进一步将 $\textbf b_n^{i}$ 定义为 $\mathcal B_n$ 中前 $i$ 层的部分和，即 $\textbf b_n = \textbf b^{\mathcal S}_n$。当 $L$ 不能被 $N$ 整除时，将最后一个部分和作为该末尾区块的表征。与 Full AttnRes 一样，$\phi$ 内部的 RMSNorm 能够防止完整区块与部分和之间的数值量级差异对注意力权重产生偏差。


**Inter-Block Attention。** 在 Full AttnRes 中，第 $l$ 层的输入是通过对截至 $f_{l-1}(h_{l-1})$ 的所有输出进行注意力计算得到的。分块版本使用块表示（Block Representations）替代这些单独的输出，并定义 $b_0 = h_1$，以确保词元嵌入（Token Embedding）始终被包含为一个信息源。对于第 $n$ 个块中的第 $i$ 层，其值矩阵（Value Matrix）为：

```math
\mathbf{V} =
\begin{cases}
[b_0, b_1, \ldots, b_{n-1}]^\top,
& \text{当 } i = 1 \text{（块 } n \text{ 的第一层）} \\[6pt]
[b_0, b_1, \ldots, b_{n-1}, b_n^{i-1}]^\top,
& \text{当 } i \geq 2 \text{（后续层）}
\end{cases}
\tag{6}
```

键（Keys）和注意力权重（Attention Weights）的计算遵循公式（3）和公式（2）。网络第一层的输入是词元嵌入，即 $b_0 = h_1$。在每个块中，第一层接收之前所有块的表示以及词元嵌入，而后续层还会额外关注块内部分和（Partial Sum）$b_n^{i-1}$。最终输出层聚合全部 $N$ 个块的表示。图 2 给出了 Block AttnRes 的 PyTorch 风格伪代码。

**Efficiency。** 由于每一层现在只需要关注 $N$ 个块表示，而不再是 $L$ 个独立的层输出，因此内存复杂度从 $O(L)$ 降低至 $O(N)$，计算复杂度从 $O(L^2)$ 降低至 $O(N^2)$。块数量 $N$ 可以在两个极端情况之间进行调整：当 $N = L$ 时，恢复为 Full AttnRes；当 $N = 1$ 时，退化为标准残差连接（Standard Residual Connections），其中词元嵌入被单独保留为 $b_0$。根据实验结果，我们发现，在不同模型规模下，$N \approx 8$ 就能恢复 Full AttnRes 的大部分性能收益，同时每个词元只需要存储 8 个隐藏状态（Hidden States）（参见第 5 节）。

除了内存和计算方面的优势之外，分块结构还能够改善推理延迟（Inference Latency）：块边界定义了第 3 节所述分块优化方法的任务分发粒度（Dispatch Granularity），而固定的块数量 $N$ 则为 KV Cache 的大小提供了上界。通过在线 Softmax（Online Softmax）[31]，将并行计算得到的块间结果（Inter-Block Results）与顺序计算得到的块内部分和（Intra-Block Partial Sums）进行合并，从而保证计算结果的严格等价性（Exact Equivalence）（参见第 4 节）。

## 4.Infrastructure Design

### 4.1 Training

### 4.2 Inference

## 5.Experiments
