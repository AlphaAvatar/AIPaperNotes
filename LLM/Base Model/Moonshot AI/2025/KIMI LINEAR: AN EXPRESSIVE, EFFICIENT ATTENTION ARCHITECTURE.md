论文链接：https://arxiv.org/pdf/2510.26692

代码链接：https://github.com/MoonshotAI/Kimi-Linear

# 摘要

我们提出了 Kimi Linear，一种混合​​线性注意力架构。该架构首次在各种场景（包括短上下文、长上下文和强化学习 (RL) 扩展机制）的公平比较中超越了完全注意力机制。其核心是 Kimi Delta Attention (KDA)，这是一个表达力强的线性注意力模块，它通过更细粒度的门控机制扩展了 Gated DeltaNet，从而能够更有效地利用有限的有限状态 RNN 内存。我们定制的分块算法通过 Diagonal-Plus-Low-Rank (DPLR) 转移矩阵的特殊变体实现了高硬件效率，与通用 DPLR 公式相比，该变体在保持与经典 delta 规则更一致的同时，显著降低了计算量。

我们预训练了一个 Kimi Linear 模型，该模型具有 3B 激活参数和 48B 总参数，基于 KDA 和多头潜在注意力 (MLA) 的逐层混合模型。实验表明，在相同的训练方案下，Kimi Linear 在所有评估任务中均显著优于完整的 MLA 模型，同时将 key-value 缓存使用量降低了高达 75%，并在 1M 上下文中实现了高达 6 倍的解码吞吐量提升。这些结果表明，Kimi Linear 可以作为完整注意力架构的直接替代方案，并具有更优异的性能和效率，包括输入输出长度更长的任务。

为了支持进一步的研究，我们开源了 KDA 内核和 vLLM 实现，并发布了预训练和指令微调的模型检查点。

# 1.介绍

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de65a58417954c74b6427489e8a820ad.png)

随着大语言模型（LLM）演化为功能日益强大的智能体，推理的计算需求——尤其是在长时域和强化学习（RL）场景下——正成为一个核心瓶颈。这种向强化学习测试时扩展的转变，使得模型必须在推理时处理扩展轨迹、工具使用交互以及复杂的决策空间，从而暴露了标准注意力机制的根本性缺陷。特别是，softmax 注意力机制的二次时间复杂度和线性增长的键值（KV）缓存引入了大量的计算和内存开销，阻碍了吞吐量、上下文长度扩展和实时交互能力。

线性注意力机制提供了一种降低计算复杂度的原则性方法，但由于其表达能力有限，在语言建模方面，即使是对于短序列，其性能也一直逊于 softmax 注意力机制。近年来，通过两项创新显著缩小了这一差距：**门控或衰减机制**和 **delta 规则**。这些进展共同推动线性注意力机制在中等长度序列上的性能更接近softmax水平。然而，纯粹的线性结构仍然受到有限状态容量的根本限制，这使得长序列建模和上下文检索在理论上仍然具有挑战性。

结合 softmax 和线性注意力机制的混合架构——即在速度更快的线性层之外，使用少量全局注意力层——已成为质量和效率之间一种切实可行的折衷方案。然而，以往的混合模型通常规模有限，或者缺乏在各种基准测试上的全面评估。**核心挑战依然存在**：开发一种在质量上能够达到甚至超越完全注意力机制，同时在速度和内存方面实现显著效率提升的注意力架构——这是实现下一代智能体、解码密集型LLM的关键一步。

本文提出了一种名为 Kimi Linear 的混合线性注意力架构，旨在满足智能体智能的效率需求和测试时间扩展性，同时又不牺牲模型质量。其核心是 **Kimi Delta Attention (KDA)**，这是一个（1）**硬件高效**的线性注意力模块，它在 Gated DeltaNet 的基础上扩展了一种更细粒度的门控机制。与 GDN（类似于 Mamba2）采用粗粒度的头部遗忘门控不同，（2）**KDA 引入了一种通道级变体，其中每个特征维度都保持着独立的遗忘率**，类似于 Gated Linear Attention (GLA)。这种细粒度设计能够更精确地控制有限状态 RNN 的记忆，从而释放混合架构中 RNN 类模型的潜力。

至关重要的是，KDA 使用 Diagonal-Plus-Low-Rank (DPLR) 矩阵的特殊变体对其转移动态进行参数化，从而实现定制的分块并行算法，该算法相对于一般的 DPLR 公式大幅减少了计算量，同时保持与经典 delta 规则的一致性。

Kimi Linear 将 KDA 与周期性的全注意力层以 3:1 的均匀比例交错排列。这种混合结构在生成长序列时，通过全注意力层保持全局信息流，同时将内存和键值缓存的使用量降低高达 75%。通过匹配规模的预训练和评估，我们证明 Kimi Linear 在短上下文、长上下文和强化学习风格的后训练任务中，始终能够达到或超越强大的全注意力基线模型的性能——同时在 100 万上下文长度下，解码吞吐量最高可提升 6 倍。

为了促进进一步的研究，我们发布了集成了 vLLM 的开源 KDA 内核，以及预训练和指令微调的检查点。这些组件与现有的全注意力流水线无缝兼容，无需修改缓存或调度接口，从而有助于混合架构的研究。

**Contributions**
- **Kimi Delta Attention (KDA)**：一种线性注意力机制，通过改进循环记忆管理和硬件效率来完善门控 delta 规则。
- **The Kimi Linear architecture**：采用 3:1 KDA 与全局注意力比率的混合设计，在减少内存占用的同时超越了完全注意力质量。
- **Fair empirical validation at scale**：通过 1.4T 个 token 的训练运行，Kimi Linear 在短/长上下文和 RL 风格的评估中优于完全注意力机制和其他基线，并完全释放了内核、集成了 vLLM 和检查点。

# 2.Preliminary

在本节中，我们将介绍与我们提出的 Kimi Delta Attention 相关的技术背景。

## 2.1 Notation

本文中，我们定义 $□_t ∈ \mathbb R^{d_k}~or~\mathbb R^{d_v}$，使得 $□ ∈ \{q, k, v, o, u, w\}$ 表示对应的第 $t$ 个列向量，$S_t ∈ \mathbb R^{d_k×d_v}$ 表示矩阵形式的存储状态。$\textbf M$ 和 $\textbf M^−$ 分别表示带对角元素和不带对角元素的下三角掩码；为方便起见，我们也将其分别记为 $Tril$ 和 $StrictTril$。

**Chunk-wise Formulation**。假设序列被分割成长度为 $L/C$ 的块，每个块的长度为 $C$。我们定义 $□_{[t]} ∈ \mathbb R^{C×d}$，其中 $□ ∈ \{\textbf Q, \textbf K, \textbf V, \textbf O, \textbf U, \textbf W\}$ 是堆叠第 $t$ 个块内向量的矩阵，$□^r_{[t]} = □_{tC+r}$ 是该块的第 $r$ 个元素。注意，$t ∈ [0, L/C), r ∈ [1, C]$。状态矩阵也进行了重新索引，使得 $S^i_{[t]} = S_{tC+i}$。此外，$S_{[t]}:= S^0_{[t]} = S^C_{[t−1]}$，即，一个块的初始状态是前一个块的最后一个状态。

**Decay Formulation**。我们定义累积衰减（cumulative decay）为 ${\color{red}{\gamma^{i \rightarrow j}_{[t]}}} := \prod_{k=i}^{j} {\color{red}{\alpha^k_{[t]}}}$ 并将 $\color{red}{\gamma^{1 \rightarrow r}_{[t]}}$ 简写为 $\color{red}{\gamma^{r}_{[t]}}$。此外，${\color{red}{\mathcal{A}_{[t]} := \mathcal{A}^{i/j}_{[t]}}} \in \mathbb{R}^{C \times C}$ 是一个矩阵，其元素为 $\frac{\color{red}{\gamma^{i}_{[t]}}}{\color{red}{\gamma^{j}_{[t]}}}$。$\color{red}{\mathrm{Diag}(\textbf{α}_t)}$ 表示**细粒度衰减（fine-grained decay）**，并且 ${\color{red}{\mathrm{Diag}\big(\textbf{γ}^{i \rightarrow j}_{[t]}\big)}} := \prod_{k=i}^{j} {\color{red}{\mathrm{Diag}\big(\textbf{α}^k_{[t]}\big)}}$。最后，${\color{red}{\Gamma^{i \rightarrow j}_{[t]}}} \in \mathbb{R}^{C \times d_k}$ 表示从 $\color{red}{\textbf{γ}^{i}_{[t]}}$ 到 $\color{red}{\textbf{γ}^{j}_{[t]}}$ 的矩阵堆叠（matrix stack）。

## 2.2 Linear Attention and the Gated Delta Rule

**Linear Attention as Online Learning**。线性注意力维护一个矩阵值循环状态，该状态会累积 key-value 关联：

$$\textbf S_t=\textbf S_{t-1}+\textbf k_t\textbf v^⊤_t,\quad \textbf o_t=\textbf S^⊤_t\textbf q_t.$$

从快速权重的角度来看，$\textbf S_t$ 充当关联存储器，**存储从 key 到 value 的瞬态映射**。这种更新可以看作是对**无界相关性目标函数**执行梯度下降。

$$\mathcal L_t(\textbf S)=-⟨\textbf S^⊤\textbf k_t, \textbf v_t⟩,$$

这种方法会不断强化最近的键值对，而不会遗忘任何信息。然而，这种目标并没有提供删除哪些记忆的标准，累积状态会无限增长，导致对长期上下文的干扰。

**DeltaNet: Online Gradient Descent on Reconstruction Loss**。DeltaNet 将这种重复重新解释为在重建目标上的在线梯度下降：

$$\mathcal L_t(\textbf S)=\frac{1}{2}||\textbf S^⊤\textbf k_t-\textbf v_t||^2$$

以学习率 $β_t$ 进行梯度更新，结果如下：

$$\textbf S_t=\textbf S_{t-1}-\beta_t∇_{\textbf S}\mathcal L_t(\textbf S_{t-1})=(\textbf I-\beta_t\textbf k_t\textbf k^⊤_t)\textbf S_{t-1}+\beta_t\textbf k_t\textbf v^⊤_t.$$

这条规则——经典的 delta 规则——将 $\textbf S$ 视为一个可学习的联想记忆，它不断地自我纠正，朝着映射 $\textbf k_t \mapsto \textbf v_t$ 的方向发展。秩 1 更新结构等价于广义的 Householder 变换，支持硬件高效的分块并行化。


**Gated DeltaNet as Weight Decay**。尽管 DeltaNet 能够稳定学习过程，但它仍然会无限期地保留过时的关联。Gated DeltaNet (GDN) 引入了一个标量遗忘门 ${\color{red}{α}_t} ∈ [0, 1]$，从而得到：

$$\textbf S_t={\color{red}{α}_t} (\textbf I-\beta_t\textbf k_t\textbf k^⊤_t)\textbf S_{t-1}+\beta_t\textbf k_t\textbf v^⊤_t.$$

在此，$\color{red}{\alpha_t}$ 起到对快速权重进行权重衰减的作用，实现了一种类似于数据相关的 L2 正则化的遗忘机制。这种简单而有效的改进提供了一种控制记忆寿命和减轻干扰的合理方法，在保持 DeltaNet 并行化结构的同时，提高了模型的稳定性和长上下文泛化能力。

从这个角度来看，我们观察到 GDN 可以解释为一种乘法位置编码形式，其中转移矩阵是数据相关的且可学习的，从而放宽了 RoPE 的正交性约束。（当状态变换矩阵保持其正交性时，也可以将绝对位置编码独立地应用于 $\textbf q$ 和 $\textbf k$，以便在注意力计算期间转换为相对位置编码。）

# 3.Kimi Delta Attention: Improving Delta Rule with Fine-grained Gating

我们提出了一种新的门控线性注意力机制——Kimi Delta Attention (KDA)。KDA通过引入细粒度的对角化门 $\color{red}Diag(\textbf α_t)$ 改进了GDN的标量衰减机制，从而能够对记忆衰减和位置感知进行精细控制（详见§6.1）。首先，我们介绍了 KDA 的分块并行化，展示了如何在对角门控下保持稳定性的同时，将一系列秩为1的矩阵变换压缩成稠密表示。然后，我们重点阐述了 KDA 相对于标准DPLR（对角加低秩）公式的效率优势。

$$\textbf S_t=(\textbf I-\beta_t\textbf k_t\textbf k^⊤_t){\color{red}Diag(\textbf α_t)}\textbf S_{t-1}+\beta_t\textbf k_t\textbf v^⊤_t\in\mathbb R^{d_k\times d_v};\quad \textbf o_t=\textbf S^⊤_t\textbf q_t\in\mathbb R^{d_v}\tag{1}$$

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f3ea42dd53241cdb5f291a54fb8f2cc.png)

## 3.1  Hardware-Efficient Chunkwise Algorithm

通过将式 (1) 的递推关系部分展开为分块形式，我们得到：

$$\mathbf{S}^{r}_{[t]} =
\underbrace{
\left(
\prod_{i=1}^{r}
\left(
\mathbf{I} - \beta^{i}_{[t]} \mathbf{k}^{i}_{[t]} \mathbf{k}^{i\top}_{[t]}
\right)
\operatorname{Diag}(\alpha^{i}_{[t]})
\right)
}_{:=\mathbf{P}^{r}_{[t]}}
\mathbf{S}^{0}_{[t]}
+
\underbrace{
\sum_{i=1}^{r}
\left(
\prod_{j=i+1}^{r}
\left(
\mathbf{I} - \beta^{j}_{[t]} \mathbf{k}^{j}_{[t]} \mathbf{k}^{j\top}_{[t]}
\right)
\operatorname{Diag}(\alpha^{j}_{[t]})
\right)
\beta^{i}_{[t]} \mathbf{k}^{i}_{[t]} \mathbf{v}^{i\top}_{[t]}
}_{:=\mathbf{H}^{r}_{[t]}}
\tag{2}$$

**WY Representation**。WY 表示通常用于将一系列秩为 1 的更新打包为一个紧凑的表示形式。我们遵循 Comba 中对 $\mathbf{P}$ 的公式化方式，以减少后续计算中额外矩阵求逆的需求。

$$\mathbf{P}^{r}_{[t]} =
{\color{red}\operatorname{Diag}(\gamma^{r}_{[t]})}
-\sum_{i=1}^{r}
{\color{red}\operatorname{Diag}(\gamma^{i \rightarrow r}_{[t]})}
\mathbf{k}^{i}_{[t]} \mathbf{w}^{i\top}_{[t]}
\qquad
\mathbf{H}^{r}_{[t]} =
\sum_{i=1}^{r}
{\color{red}\operatorname{Diag}(\gamma^{i \rightarrow r}_{[t]})}
\mathbf{k}^{i}_{[t]} \mathbf{u}^{i\top}_{[t]}
\tag{3}$$

其中，辅助向量 $\mathbf{w}_t \in \mathbb{R}^{d_k}$ 和 $\mathbf{u}_t \in \mathbb{R}^{d_v}$ 通过如下递推关系计算得到：

$$\mathbf{w}^{r}_{[t]} =
\beta^{r}_{[t]}
\left(
{\color{red}\operatorname{Diag}(\gamma^{r}_{[t]})} \mathbf{k}^{r}_{[t]}
\sum_{i=1}^{r-1}
\mathbf{w}^{i}_{[t]}
\left(
\mathbf{k}^{i\top}_{[t]}
{\color{red}\operatorname{Diag}(\gamma^{i \rightarrow r}_{[t]})}
\mathbf{k}^{r}_{[t]}
\right)
\right)
\tag{4}$$

$$\mathbf{u}^{r}_{[t]} =
\beta^{r}_{[t]}
\left(
\mathbf{v}^{r}_{[t]}
\sum_{i=1}^{r-1}
\mathbf{u}^{i}_{[t]}
\left(
\mathbf{k}^{i\top}_{[t]}
{\color{red}\operatorname{Diag}(\gamma^{i \rightarrow r}_{[t]})}
\mathbf{k}^{r}_{[t]}
\right)
\right)
\tag{5}$$

**UT transform**。我们应用 UT 变换来减少非矩阵乘法（non-matmul）的 FLOPs，这对于在训练过程中实现更好的硬件利用率至关重要。

$$\mathbf{M}_{[t]} =
\left(
\mathbf{I}
+
\operatorname{StrictTril}
\left(
\operatorname{Diag}(\beta_{[t]})
({\color{red}\Gamma^{1 \rightarrow C}_{[t]} }\odot \mathbf{K}_{[t]})
\left(
\frac{\mathbf{K}_{[t]}}{{\color{red}\Gamma^{1 \rightarrow C}_{[t]}}}
\right)^{\top}
\right)
\right)^{-1}
\operatorname{Diag}(\beta_{[t]})
\tag{6}$$

$$\mathbf{W}_{[t]} = \mathbf{M}_{[t]} ({\color{red}\Gamma^{1 \rightarrow C}_{[t]}} \odot \mathbf{K}_{[t]}),
\qquad
\mathbf{U}_{[t]} = \mathbf{M}_{[t]} \mathbf{V}_{[t]}
\tag{7}$$

下三角矩阵的逆可以通过高斯消元中的前向代入（forward substitution）以逐行迭代的方式高效计算。

等价地，在矩阵形式下，我们可以以分块方式更新状态：

$$\mathbf{S}_{[t+1]} =
{\color{red}\operatorname{Diag}(\gamma^{C}_{[t]})} \mathbf{S}_{[t]}
+
({\color{red}\Gamma^{1 \rightarrow C}_{[t]}} \odot \mathbf{K}_{[t]})^{\top}
(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]})
\in \mathbb{R}^{d_k \times d_v}
\tag{8}$$

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/109783b14c1a4ac787f8f118e0a4ef36.png)

在输出阶段，我们采用块间递归和块内并行策略来最大化矩阵乘法吞吐量，从而充分利用张量核心的计算潜力。

$$\mathbf{O}_{[t]} =
\underbrace{
({\color{red}\Gamma^{1 \rightarrow C}_{[t]}} \odot \mathbf{Q}_{[t]}) \mathbf{S}_{[t]}
}_{\text{inter-chunk}}
+
\underbrace{
\operatorname{Tril}
\left(
({\color{red}\Gamma^{1 \rightarrow C}_{[t]}} \odot \mathbf{Q}_{[t]})
\left(
\frac{\mathbf{K}_{[t]}}{{\color{red}\Gamma^{1 \rightarrow C}_{[t]}}}
\right)^{\top}
\right)
}_{\text{intra-chunk}}
(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]})
\in \mathbb{R}^{C \times d_v}
\tag{9}$$

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4a6b83d9814548158a342d6a8519f88c.png)


## 3.2 Efficiency Analysis

从表示能力的角度来看，KDA 与广义 DPLR 形式是一致的，即 $\mathbf{S}_t = (\mathbf{D} - \mathbf{a}_t \mathbf{b}_t^{\top}) \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^{\top}$，二者都表现出细粒度的衰减行为。然而，这种细粒度衰减在执行除法运算时会引入数值精度问题（例如式 (9) 中的块内计算）。为了解决这一问题，先前的工作（如 GLA）在对数域中进行计算，并在全精度下引入二级分块。然而，这种方法阻碍了半精度矩阵乘法的充分利用，并显著降低了算子速度。通过将变量 (a) 和 (b) 同时绑定到 (k)，KDA 有效缓解了这一瓶颈——将二级分块矩阵计算的数量从四次减少到两次，并进一步消除了额外的三次矩阵乘法。因此，与 DPLR 形式相比，KDA 的算子效率提升了约 100%。更为详细的分析见 §6.2。

# 4.The Kimi Linear Model Architecture

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/336fd99326d34b5a9bd6e0fbad730a1a.png)


我们的模型架构主要基于 Moonlight。除了细粒度的门控之外，我们还利用了多个组件来进一步提升 Kimi Linear 的表达能力。Kimi Linear 的整体架构如图 3 所示。

**Neural Parameterization**。设 $x_t \in \mathbb{R}^d$ 为第 $t$ 个 token 的输入表示，则每个头 $h$ 的 KDA 输入计算如下：

$$\mathbf{q}_t^h, \mathbf{k}_t^h
= \operatorname{L2Norm}\big(\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_{qk}^h x_t))\big)
\in \mathbb{R}^{d_k}$$

$$\mathbf{v}_t^h
= \operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_v^h x_t))
\in \mathbb{R}^{d_v}$$

$$\boldsymbol{\alpha}_t^h
= f(\mathbf{W}_\alpha^{↑} \mathbf{W}_\alpha^↓ x_t)
\in [0,1]^{d_k}$$

$$\beta_t^h
= \operatorname{Sigmoid}(\mathbf{W}_\beta^h x_t)
\in [0,1]$$

其中，$d_k, d_v$ 分别表示 key 和 value 的头维度，在所有实验中均设为 128。对于 $q, k, v$，我们先应用 ShortConv，然后接 Swish 激活函数，遵循 [111]。随后，对 $q$ 和 $k$ 使用 L2Norm 进行归一化，以确保特征值稳定性，如 [112] 所建议的那样。逐通道的衰减项 $\boldsymbol{\alpha}_t^h$ 通过低秩投影进行参数化（$\mathbf{W}_\alpha^↓$ 与 $\mathbf{W}_\alpha^{↑}$ 的秩等于头维度），并结合一个与以往工作中类似的衰减函数 $f(\cdot)$，这遵循 GDN 和 Mamba 的设置。在通过 $\mathbf{W}_o \in \mathbb{R}^{d \times d}$ 进行输出投影之前，我们使用**按头的 RMSNorm**，并引入一种**数据依赖的门控机制**，其参数化形式如下：

$$\mathbf{o}_t=
\mathbf{W}_o
\Big(
\operatorname{Sigmoid}(\mathbf{W}_g^↑ \mathbf{W}^↓_g x_t)
\odot
\operatorname{RMSNorm}(\operatorname{KDA}(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \boldsymbol{\alpha}_t, \beta_t))
\Big)
\tag{10}$$

其中，输出门控采用低秩参数化方式，类似于遗忘门（forget gate），以确保在参数量上与全秩门控方式进行公平比较，同时在保持性能相当的情况下缓解 Attention Sink 问题。非线性激活函数的选择将在 §5.2 中进一步讨论。

**Hybrid model architecture**。对于纯线性注意力机制而言，长上下文检索仍然是主要瓶颈，因此我们将KDA与少量全全局注意力（Full MLA）层混合使用。对于Kimi Linear，我们选择逐层方法（交替使用整层）而非逐头方法（在层内混合使用不同的注意力头），因为前者具有更简洁的架构和更高的训练稳定性。经验表明，3:1的统一比例，即3个KDA层对应1个全MLA层，能够提供最佳的质量-吞吐量平衡。我们将在7.2节讨论其他混合策略。

**No Position Encoding (NoPE) for MLA Layers**。在 Kimi Linear 中，我们将 NoPE 应用于所有全注意力（MLA）层。这种设计将编码位置信息和近因偏好（参见 § 6.1）的全部责任委托给了 KDA 层。**因此，KDA 被确立为主要的位置感知算子，其作用类似于，甚至可能比短卷积或 SWA 等辅助组件更强**。我们的发现与先前的研究结果 [110, 7, 19] 一致，这些研究同样表明，用专门的位置感知机制来补充全局 NoPE 注意力可以获得具有竞争力的长上下文性能。

# 5.Experiments