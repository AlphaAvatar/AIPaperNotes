# mHC: Manifold-Constrained Hyper-Connections

论文链接：https://arxiv.org/pdf/2512.24880

代码链接：

## 摘要

近年来，以超连接（Hyper-Connections，HC）为代表的研究通过扩展残差流宽度和多样化连接模式，扩展了过去十年广泛应用的残差连接范式。虽然这种多样化带来了显著的性能提升，但它从根本上损害了**残差连接固有的恒等映射特性**，导致严重的训练不稳定和可扩展性受限，并增加了显著的内存访问开销。为了应对这些挑战，我们提出了 **Manifold-Constrained Hyper-Connections (mHC)**。mHC 是一个通用框架，它将 HC 的残差连接空间投影到特定的流形上以恢复恒等映射特性，同时结合严格的基础设施优化来确保效率。实验表明，mHC 能够有效地进行大规模训练，提供显著的性能提升和优异的可扩展性。我们预期，作为 HC 的灵活实用扩展，mHC 将有助于加深对拓扑架构设计的理解，并为基础模型的演进指明有前景的方向。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/35c990e22ccf4da590e10cf637709335.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

自 ResNet 提出以来，深度神经网络架构经历了快速发展。如图 1(a) 所示，单层结构可以表述如下：

```math
\textbf x_{l+1}=\textbf x_l+\mathcal F(\textbf x_l,\mathcal W_l),\tag{1}
```

其中 $\textbf x_l$ 和 $\textbf x_{l+1}$ 分别表示第 $l$ 层的 C 维输入和输出，$F$ 表示残差函数。尽管残差函数 $F$ 在过去十年中不断发展，融合了卷积、注意力机制和前馈网络等多种操作，但残差连接的范式依然保持着其原始形式。随着 Transformer 架构的发展，这一范式如今已成为大语言模型 (LLM) 的基本设计元素。

这一成功主要归功于残差连接的简洁形式。更重要的是，早期研究表明，残差连接的**恒等映射特性**在大规模训练过程中能够保持稳定性和效率。通过将残差连接递归地扩展到多个层，公式（1）可得：

```math
\textbf x_L=\textbf x_l+\sum^{L-1}_{i=l}\mathcal F(\textbf x_i,\mathcal W_i),\tag{2}
```

其中 $L$ 和 $l$ 分别对应于更深层和更浅层。术语“**恒等映射**”指的是分量 $\textbf x_l$ 本身，它强调了来自较浅层的信号无需任何修改即可直接映射到更深层这一特性。

近年来，以 Hyper-Connections (HC) 为代表的研究为残差连接引入了新的维度，并通过实验验证了其性能潜力。图 1(b) 展示了 HC 的单层架构。通过扩展残差流的宽度并增强连接复杂度，HC 在不改变单个单元的浮点运算开销的情况下，显著提高了拓扑复杂度。形式上，HC 中的单层传播定义为：

```math
\textbf x_{l+1}=\mathcal H^{res}_l\textbf x_l+\mathcal {H^{post}_l}^T\mathcal F(\mathcal H^{pre}_l\textbf x_l,\mathcal W_l),\tag{3}
```

其中 $\textbf x_l$ 和 $\textbf x_{l+1}$ 分别表示第 $l$ 层的输入和输出。与公式 (1) 中的公式不同，$\textbf x_l$ 和 $\textbf x_{l+1}$ 的特征维度从 $C$ 扩展到 $n × C$，其中 $n$ 为扩展率。项 $\mathcal H^{res}_l∈ \mathbb R^{n×n}$ 表示一个可学习的映射，用于混合残差流中的特征。同样，作为一个可学习的映射，$\mathcal H^{pre}_l∈ \mathbb R^{1×n}$ 将来自 $𝑛𝐶$ 维流的特征聚合到一个 $C$ 维的层输入中，反之，$\mathcal H^{post}_l∈ \mathbb R^{1×n}$ 将层输出映射回流。

然而，随着训练规模的增大，$HC$ 引入了潜在的不稳定性风险。**主要问题在于，当架构跨越多层时，HC 的非约束特性会损害恒等映射性质**。在包含多个并行流的架构中，理想的恒等映射起到了守恒机制的作用。它确保在正向和反向传播过程中，各流的平均信号强度保持不变。通过公式 (3) 递归地将 HC 扩展到多层，可得到：

```math
\textbf x_L=\left(\prod^{L-1}_{i=1}\mathcal H^{res}_{L-i}\right)\textbf x_l+\sum^{L-1}_{i=l}\left(\prod^{L-1-i}_{j=1}\mathcal H^{res}_{L-h}\right)\mathcal{H^{post}_i}^T\mathcal F(\mathcal H^{pre}_i\textbf x_i,\mathcal W_i),\tag{4}
```

其中，$L$ 和 $l$ 分别代表更深层和更浅层。与公式 (2) 不同，HC 中的复合映射 $\prod^{L-l}_{i=1} \mathcal H^{res}_{L−i}$ 无法保持特征的全局均值。这种差异会导致信号放大或衰减无界，从而在大规模训练过程中造成不稳定。此外，虽然 HC 在 FLOPs 方面保持了计算效率，但原始设计并未解决扩展残差流的内存访问成本等硬件效率问题。这些因素共同限制了 HC 的实际可扩展性，并阻碍了其在大规模训练中的应用。

为了应对这些挑战，我们提出了 **Manifold-Constrained Hyper-Connections (mHC)**，如图 1(c) 所示。mHC 是一个通用框架，它将超连接（HC）的残差连接空间投影到特定的流形上，以恢复恒等映射性质，同时结合严格的基础设施优化来确保效率。具体而言，mHC 利用 **Sinkhorn-Knopp** 算法将 $\mathcal H^{res}_l$ 熵投影到 Birkhoff 多面体上。此操作有效地将残差连接矩阵约束在由双随机矩阵构成的流形内。由于这些矩阵的行和列和均为 1，因此 $\mathcal H^{res}_𝑙\textbf x_l$ 操作相当于输入特征的凸组合。这一特性有利于信号传播的良好条件，其中特征均值保持不变，信号范数得到严格正则化，从而有效地降低了信号消失或爆炸的风险。此外，由于**双随机矩阵的矩阵乘法封闭，复合映射 $\prod^{L−l}_{i=1}\mathcal H^{res}_{L−i}$保留了这种守恒性质**。因此，mHC 有效地维护了任意深度之间恒等映射的稳定性。为了确保效率，我们采用了核融合技术，并利用 TileLang 开发了混合精度核。此外，我们通过选择性重计算和在 DualPipe 调度中精心安排通信重叠来减少内存占用。

大量的语言模型预训练实验表明，mHC 在保持 HC 性能优势的同时，展现出卓越的稳定性和可扩展性。内部大规模训练表明，mHC 支持大规模训练，并且在扩展率 $n = 4$ 时仅引入 6.7% 的额外时间开销。

## 2.Related Works

深度学习的**架构进步**主要可分为**微观设计**和**宏观设计**。（1）微观设计关注计算模块的内部架构，具体规定了特征如何在空间、时间和通道维度上进行处理（**特征如何被处理**）。（2）与之相对，宏观设计则构建模块间的拓扑结构，从而决定特征表示如何在不同层之间传播、路由和合并（**特征如何被传播**）。

### 2.1 Micro Design

卷积运算最初凭借参数共享和平移不变性主导了结构化信号的处理。虽然后续的变体，例如深度可分离卷积和分组卷积，优化了效率，但 Transformer 的出现确立了注意力机制和前馈网络（FFN）作为现代架构基本构建模块的地位。注意力机制促进全局信息传播，而 FFN 增强了单个特征的表征能力。为了平衡性能与 LLM 的计算需求，注意力机制已经发展出高效的变体，例如 Multi-Query Attention (MQA)、 Grouped-Query Attention (GQA) 和 Multi-Head Latent Attention (MLA)。与此同时，FFN 通过混合专家模型（MoE）被推广到稀疏计算范式中，从而实现了大规模的参数扩展而无需增加相应的计算成本。

### 2.2 Macro Design

宏观设计决定了网络的全局拓扑结构。继 ResNet 之后，诸如 DenseNet 和 FractalNet 等架构分别旨在通过增加密集连接和多路径结构来提升拓扑复杂性，从而提高性能。Deep Layer Aggregation (DLA) 通过递归聚合不同深度和分辨率的特征，进一步扩展了这一范式。

近年来，宏观设计的重点已转向扩展残差流的宽度。Hyper-Connections (HC) 引入了可学习矩阵来调节不同深度特征之间的连接强度，而 Residual Matrix Transformer (RMT) 则用外积记忆矩阵取代了标准的残差流，以方便特征存储。类似地，MUDDFormer 采用多路动态密集连接来优化跨层信息流。尽管这些方法具有潜力，但它们**会损害残差连接固有的恒等映射特性**，从而导致系统不稳定并阻碍可扩展性。此外，由于特征宽度的扩展，它们还会带来显著的内存访问开销。本文提出的 mHC 在 HC 的基础上，将残差连接空间限制在特定的流形上以恢复恒等映射特性，同时还融入了严格的基础设施优化以确保效率。这种方法在保持扩展连接拓扑优势的同时，增强了系统的稳定性和可扩展性。

## 3.Preliminary

首先，我们明确本文所用的符号。在 HC 模型中，第 $l$ 层的输入 $\textbf x_l\in\mathbb R^{1\times C}$ 通过乘以因子 $n$，构建隐藏矩阵 $\textbf x_l=(\textbf x^⊤_{l,0}, ..., \textbf x^⊤_{l,n-1})^⊤∈\mathbb R^{n×C}$，该矩阵可视为 $n$ 流残差。此操作有效地拓宽了残差流的宽度。为了控制该残差流的读取、写入和更新过程，HC 引入了三个可学习的线性映射——$\mathcal H^{pre}, \mathcal H^{post}∈\mathbb R^{1×n}$ 和 $\mathcal H^{res}∈\mathbb R^{n×n}$。这些映射修改了式 (1) 所示的标准残差连接，从而得到式 (3) 所示的公式。

在 HC 公式中，可学习映射由两部分系数组成：输入相关的系数和全局系数，分别称为动态映射和静态映射。形式上，HC 按如下方式计算系数：

```math
\begin{cases}
\tilde{\textbf x}_l=RMSNorm(\textbf x_l)\\
\mathcal H^{pre}_l=\alpha^{pre}_l\cdot tanh(\theta^{pre}_l\tilde{\textbf x}^T_l)+\textbf b^{pre}_l\\
\mathcal H^{post}_l=\alpha^{post}_l\cdot tanh(\theta^{post}_l\tilde{\textbf x}^T_l)+\textbf b^{post}_l\\
\mathcal H^{res}_l=\alpha^{res}_l\cdot tanh(\theta^{res}_l\tilde{\textbf x}^T_l)+\textbf b^{res}_l,
\end{cases}\tag{5}
```

其中，RMSNorm(·) 应用于最后一个维度，标量 $\alpha^{pre}_l$、$\alpha^{post}_l$ 和 $\alpha^{res}_l∈\mathbb R$ 是可学习的门控因子，初始值较小。动态映射通过线性投影导出，参数为 $\theta^{pre}_l, \theta^{post}_l∈\mathbb R^{1×C}$ 和 $\theta^{res}_l∈\mathbb R^{n×C}$，而静态映射由可学习的偏置 $\textbf b^{pre}_l, \textbf b^{post}_l∈\mathbb R^{1×b}$ 和 $\textbf b^{res}_l∈\mathbb R^{n×n}$ 表示。

值得注意的是，引入这些映射—— $\mathcal H^{pre}_l, \mathcal H^{post}_l$ 和 $\mathcal H^{res}_l$ ——几乎不会增加计算开销，因为典型的扩展率 $n$（例如 4）远小于输入维度 $C$。通过这种设计，HC 有效地将残差流的信息容量与层的输入维度解耦，而输入维度与模型的计算复杂度（FLOPs）密切相关。因此，HC 提供了一种通过调整残差流宽度来实现扩展的新途径，是对预训练缩放率中讨论的传统模型 FLOPs 和训练数据大小扩展维度的补充。

尽管 HC 需要三种映射来处理残差流和层输入之间的维度不匹配问题，但表 1 所示的初步实验表明，残差映射 $\mathcal H^{res}_l$ 能够带来最显著的性能提升。这一发现凸显了残差流内有效信息交换的关键性。

<img
  src="https://i-blog.csdnimg.cn/direct/9e129fa27e564392b995e0bad710ab01.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

### 3.1 Numerical Instability

<img
  src="https://i-blog.csdnimg.cn/direct/f758946166694502849798f7d06c99ce.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

虽然残差映射 $\mathcal H^{res}_l$ 对性能至关重要，但其顺序应用却会给数值稳定性带来重大风险。如公式 (4) 所示，当 HC 扩展到多层时，从层 $l$ 到层 $L$ 的有效信号传播由复合映射 $\prod^{L−l}_{i=1}\mathcal H^{res}_{L−i}$ 控制。由于可学习映射 $\mathcal H^{res}_l$ 不受约束，该复合映射不可避免地偏离恒等映射。因此，在正向传播和反向传播过程中，信号幅度都容易出现爆炸或消失的情况。这种现象破坏了残差学习的基本前提——信号流的畅通无阻，从而导致更深层或更大规模模型的训练过程不稳定。

经验证据支持这一分析。如图 2 所示，我们在大规模实验中观察到了不稳定的损耗行为。以 mHC 为基线，HC 在 12k 步附近出现了意料之外的损耗激增，这与梯度范数的不稳定性高度相关。此外，对 $\mathcal H^{res}_L$ 的分析验证了这种不稳定性机制。为了量化复合映射 $\prod^{L−l}_{i=1}\mathcal H^{res}_{L−i}$ 如何放大残差流上的信号，我们使用了两个指标。第一个指标基于复合映射行和的最大绝对值，捕捉了前向传播中最坏情况下的扩展。第二个指标基于列和的最大绝对值，对应于反向传播。我们将这些指标称为复合映射的最大增益幅度 (Amax Gain Magnitude)。如图 3 (b) 所示，Amax 增益幅度产生了峰值为 3000 的极端值，与 1 存在明显的偏差，这证实了爆炸残余流的存在。

<img
  src="https://i-blog.csdnimg.cn/direct/71e54036238040578df1fa7c590a013b.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

### 3.2 System Overhead

<img
  src="https://i-blog.csdnimg.cn/direct/690e3f129e8e4640995be4d612dcf5da.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

尽管由于新增映射的线性特性，HC 的计算复杂度仍然可控，但系统级开销却带来了不可忽视的挑战。具体而言，内存访问（I/O）成本通常是现代模型架构的主要瓶颈之一，这通常被称为“内存墙”。这一瓶颈在架构设计中经常被忽视，但它却对运行时效率有着决定性的影响。

本文聚焦于广泛采用的 pre-norm Transformer 架构，分析了 HC 固有的 I/O 模式。表 2 总结了 $n$ 流残差设计在单个残差层中引入的每个 token 的内存访问开销。分析表明，HC 使内存访问成本增加了一个近似与 $n$ 成正比的倍数。这种过高的 I/O 需求会显著降低训练吞吐量，除非采用融合核来缓解这一问题。此外，由于 $\mathcal H^{pre}_l$、$\mathcal H^{post}_l$ 和 $\mathcal H^{res}_l$ 涉及可学习参数，反向传播需要它们的中间激活值。这导致 GPU 内存占用大幅增加，通常需要梯度检查点来维持合理的内存使用。此外，HC 在流水线并行中需要 $n$ 倍的通信成本，导致气泡更大，从而降低训练吞吐量。

## 4.Method

### 4.1 Manifold-Constrained Hyper-Connections

受恒等映射原理的启发，mHC 的核心前提是将残差映射 $\mathcal H^{res}_l$ 约束到特定的流形上。虽然原始的恒等映射通过强制 $\mathcal H^{res}_l= \textbf I$ 来确保稳定性，但它从根本上阻碍了残差流内部的信息交换，而这对于最大化多流架构的潜力至关重要。因此，我们提出将残差映射投影到一个流形上，该流形既能保持信号在各层间传播的稳定性，又能促进残差流之间的相互作用，从而保持模型的表达能力。为此，**我们将 $\mathcal H^{res}_l$ 限制为双随机矩阵，其元素均为非负数，且行和列之和均为 1**。形式上，令 $\mathcal M^{res}$ 表示双随机矩阵的流形（也称为 Birkhoff 多面体）。我们将 $\mathcal H^{res}_l$ 约束到 $\mathcal P_{M^{res}}(\mathcal H^{res}_l)$，其定义如下：

```math
\mathcal P_{\mathcal M^{res}}(\mathcal H^{res}_l):=\{\mathcal H^{res}_l\in\mathbb R^{n\times n}|\mathcal H^{res}_l\textbf 1_n=\textbf 1_n,\textbf 1^T_n\mathcal H^{res}_l=\textbf 1^T_n,\mathcal H^{res}_l\ge 0\},\tag{6}
```

其中 $\textbf 1_n$ 表示 $n$ 维全 $1$ 向量。

值得注意的是，当 $n = 1$ 时，双重随机条件退化为标量 1，从而恢复了原始的恒等映射。双重随机性的选择赋予了模型若干严格的理论性质，有利于大规模模型的训练：
1. **Norm Preservation**。双随机矩阵的谱范数以 1 为界（即 $∥\mathcal H^{res}_l∥_2 ≤ 1$）。这意味着可学习映射是非扩张的，从而有效地缓解了梯度爆炸问题。
2. **Compositional Closure**。双随机矩阵集在矩阵乘法下封闭。这确保了跨多层的复合残差映射 $\prod^{L−l}_{i=1}\mathcal H^{res}_{L−i}$ 保持双随机性，从而在整个模型深度范围内保持稳定性。
3. **Geometric Interpretation via the Birkhoff Polytope**。集合 $\mathcal M^{res}$ 构成 Birkhoff 多面体，它是置换矩阵集合的凸包。这提供了一个清晰的几何解释：残差映射相当于置换的凸组合。从数学角度来看，重复应用此类矩阵会单调地增加跨流信息的混合程度，从而有效地发挥鲁棒特征融合机制的作用。

此外，我们对输入映射 $\mathcal H^{pre}_l$ 和输出映射 $\mathcal H^{post}_l$ 施加非负性约束。该约束可防止由**正负系数组合引起的信号抵消**，这也可以看作是一种特殊的流形投影。

### 4.2 Parameterization and Manifold Projection

本节详细介绍 mHC 中 $\mathcal H^{pre}_l$、$\mathcal H^{post}_l$ 和 $\mathcal H^{res}_l$ 的计算过程。给定第 $l$ 层的输入隐藏矩阵 $\textbf x_l ∈ \mathbb R^{n×C}$，我们首先将其展平为向量 $\vec{\mathbf{x}}_l = \mathrm{vec}(\mathbf x_l) \in \mathbb{R}^{1 \times nC}$，以保留完整的上下文信息。然后，我们按照原始 HC 公式得到动态映射和静态映射，如下所示：

```math
\begin{cases}
\vec{\mathbf{x}}_l=RMSNorm(\vec{\mathbf{x}}_l)\\
\mathcal{\tilde H}^{pre}_l=\alpha^{pre}\cdot(\vec{\mathbf{x}}'_l\varphi^{pre}_l)+\mathbf{b}^{pre}_l\\
\mathcal{\tilde H}^{post}_l=\alpha^{post}_l\cdot(\vec{\mathbf x}'_l\varphi^{post}_l)+\mathbf b^{post}_l\\
\mathcal{\tilde H}^{res}_l=\alpha^{res}_l\cdot mat(\vec{\mathbf x}'_l\varphi^{res}_l)+\textbf b^{res}_l,
\end{cases}\tag{7}
```

其中 $𝜑^{pre}_l, 𝜑^{post}_l∈ \mathbb R^{nC×n}$ 和 $𝜑^{res}_l∈ \mathbb R^{nC×n^2}$ 是动态映射的线性投影，$mat(·)$ 是从 $\mathbb R^{1×n^2}$ 到 $\mathbb R^{n×n}$ 的重塑函数。

然后，通过以下方式获得最终的约束映射：

```math
\begin{cases}
\mathcal H^{pre}_l=\sigma(\mathcal{\tilde{H}}^{pre}_l)\\
\mathcal H^{post}_l=2\sigma(\mathcal{\tilde{H}}^{post}_l)\\
\mathcal H^{res}_l=\text{Sinkhorn-Knopp}(\mathcal{\tilde{H}}^{res}_l),
\end{cases}\tag{8}
```

其中 $𝜎(·)$ 表示 Sigmoid 函数。$\text{Sinkhorn-Knopp}(·)$ 算子首先通过指数运算将所有元素变为正数，然后进行迭代归一化过程，交替地对行和列进行缩放，使之和为 1。具体来说，给定一个正矩阵 $\mathbf M^{(0)} = exp(\mathcal{\tilde H}^{res}_l)$ 作为初始值，归一化迭代过程如下：

```math
\mathbf{M}^{(t)}=\mathcal T_r\left(\mathcal T_c(\mathbf M^{(t-1)})\right ),\tag{9}
```

其中 $\mathcal T_r$ 和 $\mathcal T_c$ 分别表示行归一化和列归一化。当 $t_{max} → ∞$ 时，该过程收敛于双随机矩阵 $\mathcal H^{res}_l = \mathbf M(t_{max})$。在我们的实验中，我们选择 $t_{max} = 20$ 作为实际值。

### 4.3 Efficient Infrastructure Design

本节详细介绍专为 mHC 定制的基础设施设计。通过严格的优化，我们以仅 6.7% 的训练开销，在大规模模型中实现了 mHC（$n = 4$）。

#### 4.3.1 Kernel Fusion

观察到在对高维隐藏状态 $\vec{\mathbf x}_l ∈ \mathbb R^{1×nC}$ 进行操作时，mHC 中的 RMSNorm 会造成显著的延迟，因此我们将除以范数的操作重新排序，使其在矩阵乘法之后。这种优化在保持数学等价性的同时提高了效率。此外，我们采用混合精度策略来最大化数值精度而不牺牲速度，并将多个具有共享内存访问的操作融合到统一的计算内核中，以减少内存带宽瓶颈。基于公式 (10) 至 (13) 中详述的输入和参数，我们实现了三个专门的 mHC 内核来计算 $\mathcal H^{pre}_l$、$\mathcal H^{post}_l$ 和 $\mathcal H^{res}_l$。在这些内核中，偏差和线性投影被合并到一个 $\mathbf b_l$ 和 $𝜑_l$ 中，RMSNorm 权重也被包含在 $𝜑_l$ 中。
- 公式 (14) 至 (15)：我们开发了一个统一内核，该内核融合了对 $\vec{\mathbf x}_l$ 的两次扫描，并利用矩阵乘法单元最大限度地利用内存带宽。反向传播（包含两次矩阵乘法）也类似地被整合到一个内核中，从而避免了对 $\vec{\mathbf x}$ 的冗余重载。这两个内核都采用了经过精细调优的流水线（加载、类型转换、计算、存储），以高效地处理混合精度运算。
- 公式 (16) 至 (18)：这些对小系数的轻量级操作被巧妙地融合到一个内核中，从而显著降低了内核启动开销。
- 式 (19)：我们在单个内核中实现了 Sinkhorn-Knopp 迭代。对于反向传播，我们设计了一个自定义的反向内核，该内核在片上重新计算中间结果并遍历整个迭代过程。


```math
\begin{array}{rcllr}
\varphi_l
&:& \mathrm{tfloat32}
& [nC,\,n^2+2n]
& \text{(10)}
\\[4pt]
\vec{x}_l
&:& \mathrm{bfloat16}
& [1,\,nC]
& \text{(11)}
\\[4pt]
\alpha_l^{\mathrm{pre}},\,
\alpha_l^{\mathrm{post}},\,
\alpha_l^{\mathrm{res}}
&:& \mathrm{float32}
& \mathrm{Scalars}
& \text{(12)}
\\[4pt]
\mathbf{b}_l
&:& \mathrm{float32}
& [1,\,n^2+2n]
& \text{(13)}
\\[4pt]
\left[
\widetilde{\mathcal{H}}_l^{\mathrm{pre}},
\widetilde{\mathcal{H}}_l^{\mathrm{post}},
\widetilde{\mathcal{H}}_l^{\mathrm{res}}
\right]
&:& \mathrm{float32}
& = \vec{x}_l\varphi_l
& \text{(14)}
\\[4pt]
r
&:& \mathrm{float32}
& = \dfrac{\lVert \vec{x}_l\rVert_2}{\sqrt{nC}}
& \text{(15)}
\\[4pt]
\left[
\widetilde{\mathcal{H}}_l^{\mathrm{pre}},
\widetilde{\mathcal{H}}_l^{\mathrm{post}},
\widetilde{\mathcal{H}}_l^{\mathrm{res}}
\right]
&:& \mathrm{float32}
&
= \dfrac{1}{r}
\left[
\alpha_l^{\mathrm{pre}}
\widetilde{\mathcal{H}}_l^{\mathrm{pre}},
\alpha_l^{\mathrm{post}}
\widetilde{\mathcal{H}}_l^{\mathrm{post}},
\alpha_l^{\mathrm{res}}
\widetilde{\mathcal{H}}_l^{\mathrm{res}}
\right]
+\mathbf{b}_l
& \text{(16)}
\\[6pt]
\mathcal{H}_l^{\mathrm{pre}}
&:& \mathrm{float32}
& = \sigma\left(
\widetilde{\mathcal{H}}_l^{\mathrm{pre}}
\right)
& \text{(17)}
\\[4pt]
\mathcal{H}_l^{\mathrm{post}}
&:& \mathrm{float32}
& = 2\sigma\left(
\widetilde{\mathcal{H}}_l^{\mathrm{post}}
\right)
& \text{(18)}
\\[4pt]
\mathcal{H}_l^{\mathrm{res}}
&:& \mathrm{float32}
& = \operatorname{Sinkhorn\text{-}Knopp}
\left(
\widetilde{\mathcal{H}}_l^{\mathrm{res}}
\right)
& \text{(19)}
\end{array}
```

利用上述核函数所推导出的系数，我们进一步引入两个额外的核函数来应用这些映射：一个用于 $\mathcal{F}_{\mathrm{pre}} := \mathcal{H}_{l}^{\mathrm{pre}}\mathbf{x}_{l}$，另一个用于 $\mathcal{F}_{\mathrm{post,res}} := \mathcal{H}_{l}^{\mathrm{res}}\mathbf{x}_{l} + \mathcal{H}_{l}^{\mathrm{post}\mathsf{T}}\mathcal{F}(\cdot,\cdot)$。通过将 $\mathcal{H}_{l}^{\mathrm{post}}$ 和 $\mathcal{H}_{l}^{\mathrm{res}}$ 的应用过程与残差合并进行融合，对于该核函数，我们将读取的元素数量从 $(3n+1)C$ 减少到 $(n+1)C$，并将写入的元素数量从 $3nC$ 减少到 $nC$。除公式 $(14)$ 和 $(15)$ 外，我们使用 TileLang（Wang 等，2025）高效实现了大部分核函数。该框架简化了具有复杂计算流程的核函数实现，并使我们能够以较小的工程成本充分利用内存带宽。

#### 4.3.2 Recomputing

#### 4.3.3 Overlapping Communication in DualPipe

## 5.Experiments
