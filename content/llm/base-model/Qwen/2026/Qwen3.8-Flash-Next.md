# On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

论文链接：https://arxiv.org/pdf/2608.30320

代码链接：https://github.com/QwenLM/Qwen3.8-Flash-Next

## 摘要

本文描述了 **Qwen3.8-Flash-Next** 的架构和消融过程。Qwen3.8-Flash-Next 是一个稀疏混合专家模型，拥有 125B 参数，每个 token 激活 6B 参数，此外还有 51B 参数的 n-gram 嵌入表未在加速器上运行。在 14 个预训练基准测试中，该模型在 8 个测试中领先于其 397B-A17B 的前代模型，在其余测试中最多落后 2.6 分，激活参数数量、训练 token 数量和训练 FLOPs 数量分别约为其三分之一和九分之一。Token 混合采用了 **Gated DeltaNet (GDN)** 和全局注意力机制的逐层混合，每四个全注意力层中有一个；在持续预训练阶段，这些全注意力层被 **Qwen Sparse Attention (QSA)** 取代，QSA 使用压缩的轻量级索引器以微块粒度对上下文进行评分。残差流被扩展为四个分支，并通过逐元素门进行读取，这种设计我们称之为 **Gated Residual (GR)**。在主干网络之外，通过一个预读取主机内存的 n-gram 嵌入层来增加容量。我们从三个维度评估每个候选变更：损失及其下游基准测试结果；变更在训练、预填充和解码过程中的成本；以及变更对最优超参数和训练稳定性的影响。损失和下游准确率并非总是同步变化：扩大 n-gram 词表会单调降低损失，而下游准确率则会达到饱和。该架构和 **Muon** 优化器共同作用，提高了最优学习率和 batch size，无需 batch size 预热，并显著提升了压力测试下的稳定性。损失、基准测试结果、效率和稳定性构成了一个设计问题。通过联合解决这些问题，我们得到了一个同时更高效、更强大、更稳定的方案。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/24484991c095413d844b5c8ba1628f46.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

Qwen3.8-Flash-Next 是一个稀疏混合专家模型，总共有 125B 参数，每个 token 激活 6B 参数，此外还有 51B 参数的 n-gram 嵌入表未使用加速器处理。其设计目标是在计算资源大幅减少的情况下，保持上一代旗舰模型 397B-A17B 的性能。在涵盖知识、STEM（科学、技术、工程和数学）、推理、编程和多语言能力的 14 个预训练基准测试中，该基础模型在 8 个测试中领先于上一代模型，在其余 6 个测试中最多落后 2.6 分（表 11）。同时，每个 token 激活的参数数量和训练的 token 数量都约为上一代模型的三分之一，训练浮点运算次数 (FLOPs) 也约为上一代模型的九分之一。

架构变更会同时影响三个方面：模型在下游任务上的表现、训练和服务成本，以及训练运行在大规模应用下是否仍能保持最优性和稳定性。因此，我们从三个维度评估每一项候选变更：损失函数及其下游基准测试结果；变更在训练、预填充和解码方面的成本；以及变更对最优超参数和训练稳定性的影响。损失函数、基准测试结果、效率和稳定性构成了一个设计问题，我们将分别对每个方面进行报告。在本报告中，我们将重点阐述这三个维度之间存在的分歧，以及这些分歧最终导致的设计选择。

该设计由**四个架构组件**构成，每个组件都针对一个不同的瓶颈：
- token 混合采用了 Gated DeltaNet (GDN)  和全局注意力机制的逐层混合：循环层以线性成本将前缀压缩成固定大小的状态，而每四个循环层中就有一个全注意力层，用于保留任何有限状态记忆都无法精确复现的直接 token 级检索（§2.1.1）。
- 在持续预训练阶段，这些全注意力层被 Qwen Sparse Attention (QSA) 所取代。QSA 遵循 Liu et al. (2025a) 提出的稀疏注意力机制，但使用压缩的轻量级索引器以微块粒度对上下文进行评分，从而使索引本身的成本随序列长度的增加而降低。
- 残差流被扩展为四个分支，并通过一个逐元素门进行读取，这种设计我们称之为 Gated Residual (GR)（§2.2）：扩展增加了残差路径的容量，而门决定如何使用这些容量，同时还提供保持训练稳定性的缩放。
- 在骨干网之外，通过一个单独的 n-gram 嵌入层进一步增加容量，该嵌入层的表是从主机内存中预读取的（§2.3），从而在几乎不增加每个 token 的 FLOPs 和延迟的情况下扩展参数数量。

**Evaluation**。损失和下游准确率并非总是同步变化，我们观察到二者在两个方向上都存在差异。扩大 n-gram 词表会单调降低损失，而下游准确率则会趋于饱和；在固定参数预算下，损失最优值会偏离准确率最优值（表 9，表 8）。相反，从残差状态预测残差读写权重虽然只能略微降低损失，但却能显著提升基准性能（§2.2）。其他差异则在后期显现：将每个模块限制在两个最高门控残差分支上几乎不会影响预训练损失，但随着训练的进行，损失会逐渐降低（§2.2）；从全注意力层中移除位置编码在预训练阶段几乎没有影响，但在后期会影响生成质量（§2.1.1）。为了简洁起见，在后续章节中，当损失和基准性能同步变化时，我们仅报告损失。

**Efficiency**。我们分别评估训练、预填充和解码过程中的成本。
- 在训练阶段，FlashQLA 在 GPU 上相对于 Triton 基线实现了 2-3 倍的前向加速和大约 2 倍的后向加速（§2.1.1）。Muon 引入了自身的工程成本：其每个参数的 FLOP 数取决于矩阵形状而非参数数量，因此数据并行梯度缓冲区会根据估计的正交化成本进行重新划分；并且其步骤在融合参数分割后会分解成许多小的内核，因此该步骤会被捕获到 CUDA 图中（§3.1）。我们将牛顿-舒尔茨迭代次数设置为 8 步，以提高其在压力下的稳定性。
- 在推理阶段，预填充主要依赖于对整个上下文的注意力机制，而 QSA 通过因子 $r$ 将 key 序列压缩来解决这个问题，从而将索引器成本从 $O(n^2)$ 降低到 $O(n^2/r)$。
- 解码过程主要消耗内存，因此 GDN 层维护固定大小的循环状态，GR 层舍弃了分支混合算子 $H_{res}$，并且残差状态支持 FP8 存储。在上下文长度为 1M 的情况下，QSA 在内核级别的预填充速度比密集注意力机制快 7.6 倍，解码速度快 4.9 倍。

**Optimization**。Muon 算法应用于作为线性映射的二维权重；输入和 n-gram 嵌入、输出头、MoE 路由器以及 GR 的低秩投影仍然保留在 AdamW 上，因为正交化对于 AdamW 来说要么不切实际，要么没有帮助（§3.1）。融合参数在正交化之前被拆分，因为对连接矩阵进行正交化会将不相关子块中的奇异方向混合在一起（§3.1）。新的架构和优化器也改变了最优超参数，因此我们重新拟合了用于 Qwen3.5 系列的缩放定律（§3.2）。新的缩放定律预测了更大的 batch size 和学习率；这两个预测都分别得到了验证和确认。更大的 batch size 可以提高大规模并行吞吐量，而更大的学习率可以提高收敛速度。在训练初期逐渐增大 batch size 最终并不比从目标 batch size 开始更好，而且会增加 18.8% 的优化器步骤，因此我们不使用它（§3.2）。

**Training Stability**。我们通过提高和保持恒定的学习率，在压力测试中验证训练稳定性，这些测试旨在重现中等模型规模下的大规模不稳定性。评判标准是，新算法在相同压力下必须至少与被其替代的旧算法一样稳定。在最优学习率的四倍下，旧算法频繁出现峰值，而新算法则始终保持稳定（§3.3）。在单变量对上隔离 GR 中的门控，证实它是提升稳定性（相对于 Qwen3.5 架构）的关键因素。由于这些稳定性增强，Qwen3.8-Flash-Next 的全规模训练顺利进行，没有出现任何损失峰值或梯度范数异常波动，且无需依赖诸如 qk-clip 或 SwiGLU-clip 之类的显式裁剪方法。

**Organization**。第 2.1.1 节和第 2.1.2 节描述了 token 混合。第 2.2 节涵盖了门控残差，第 2.3 节涵盖了 n-gram 嵌入层，第 3.1 节涵盖了优化器，第 3.2 节涵盖了超参数缩放，第 3.3 节涵盖了稳定性压力测试。接下来将对基础模型和后训练模型进行评估。

## 2.Model Architecture

### 2.1 Attention

#### 2.1.1 GDN Hybrid Architecture

<img
  src="https://i-blog.csdnimg.cn/direct/c78b2f4b93834b6a90a2ec83570619a2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Motivation**。完全自注意力机制能够直接访问每个前一个 token 的内容，但其 token 混合成本随序列长度呈二次方增长，且其键值（KV）缓存在自回归生成过程中呈线性增长。滑动窗口注意力机制（SWA）用有界局部感受野取代全局访问，从而降低了计算和缓存消耗。然而，窗口外的信息只能间接地通过深度传播。这导致高效的局部处理和持久的内容相关记忆之间存在矛盾。

我们通过分层混合 **Gated DeltaNet (GDN)** 和全局注意力机制来解决这一矛盾。GDN 将前缀压缩成固定大小的循环状态，并根据当前内容更新该状态；而交错的全局注意力层则保留了直接的 token 级检索，这是任何有限状态循环记忆都难以精确实现的。

这一设计选择与表 1 中报告的架构消融结果一致。相对于全注意力 Transformer 基线模型，GDN 混合模型在 9 个选定基准测试中的 8 个上表现更佳。相对于 SWA 混合模型，它在 9 个基准测试中的 7 个上表现更强。这些结果支持采用混合设计，但它们本身并不能确定是哪个架构组件导致了每项改进。

**Gated Delta Recurrence**。线性注意力机制可以解释为一种快速加权记忆，它将键值关联存储在矩阵状态中。对于每个头部，令 $\textbf q_t, \textbf k_t ∈ \mathbb R^{d_k}$ 且 $\textbf v_t ∈ R^{d_v}$。按照实现约定，GDN 维护一个状态 $\textbf S_t ∈ \mathbb R^{d_k×d_v}$，它是原始公式中使用的状态约定的转置，并应用门控 delta 规则：

```math
\tilde{\textbf S}_{t-1}=\alpha_t\textbf S_{t-1},\tag{1}
```

```math
\textbf e_t=\textbf v_t-\tilde{\textbf S}^T_{t-1}\textbf k_t,\tag{2}
```

```math
\textbf S_t=\tilde{\textbf S}_{t-1}+\beta_t\textbf k_t\textbf e^T_t,\tag{3}
```

```math
\textbf y_t=\textbf S^T_t\textbf q_t,\tag{4}
```

其中 $α_t ∈ (0, 1)$ 是数据相关的衰减函数，$β_t ∈ (0, 1)$ 控制增量更新。等价地，

```math
\textbf S_t=\alpha_t(\textbf I-\beta_t\textbf k_t\textbf k^T_t)\textbf S_{t-1}+\beta_t\textbf k_t\textbf v^T_t.\tag{5}
```

这两个门控发挥着互补的作用。衰减门 $α_t$ 全局控制现有状态的生命周期，而 delta 门首先估计与 $\textbf k_t$ 关联的 value，然后仅写入残差误差。因此，重复或相似的 key 会更新现有关联，而不是累积无界的外部积。这种有针对性的擦除和写入操作使 GDN 区别于纯粹的加性线性注意力机制。

**GDN Parameterization**。给定归一化残差流输入$\textbf x_t ∈ \mathbb R^d$，GDN 使用学习到的投影计算内容特征，然后进行一个简短的深度可分离因果卷积：

```math
\textbf q_t=L2Norm(SiLU(ShortConv(\textbf W_q\textbf x_t))),\tag{6}
```

```math
\textbf k_t=L2Norm(SiLU(ShortConv(\textbf W_k\textbf x_t))),\tag{7}
```

```math
\textbf v_t=SiLU(ShortConv(\textbf W_v\textbf x_t)).\tag{8}
```

短卷积在信息被压缩到循环状态之前提供了一个显式的局部归纳偏置。L2 归一化限制了 $q/k$ 的幅度并稳定了 rank-1 delta 跃迁。

写入强度和衰减被参数化为

```math
\beta_t=\sigma(\textbf W_{\beta}\textbf x_t),\tag{9}
```

```math
\alpha_t=exp[-exp(\textbf A)softplus(\textbf W_{\alpha}\textbf x_t+\textbf b_{\alpha})].\tag{10}
```

在各个头部独立应用递归之后，头部输出被归一化，并通过输入相关的输出门进行调制：

```math
\textbf o_t=\textbf W_o[\sigma(\textbf W_z\textbf x_t)\odot RMSNorm(\textbf y_t)].\tag{11}
```

与使用 SiLU 输出门的原始 GDN 不同，我们在公式 (11) 中使用了有界 sigmoid 门，并在所有实验中都观察到了持续的性能提升。与 Qwen3-Next 类似，我们采用零中心 RMSNorm 来约束 RMSNorm 权重的增长。该公式同样适用于模型中使用的所有其他 RMSNorm 层。图 2 总结了完整的 GDN token 混合路径。

在模型层面，我们的混合架构在其全注意力层中保留了旋转位置嵌入（RoPE）。RoPE 和不包含位置编码的 NoPE 变体在预训练阶段表现差异不大，但 NoPE 变体在后训练阶段表现出更高的无限生成率，因此更容易无法终止。我们每四层放置一个这样的全注意力层，其余三层使用 GDN。这种分配方式在效率和质量之间取得了良好的平衡，而周期性的全注意力对于长上下文性能尤为重要。

**Kernel Efficiency**。为了提高效率，我们使用 FlashQLA（一个基于 TileLang 的融合线性注意力内核库）优化了 GDN 内核。在 NVIDIA GPU 的多种设置下，FlashQLA 相比 FLA Triton 内核实现了 2-3 倍的前向加速和约 2 倍的后向加速。实现和基准测试结果可在 https://github.com/QwenLM/FlashQLA 获取。

**Architecture Ablation**。我们使用相同的评估流程对三个检查点进行评估：全注意力 Transformer 模型、SWA 混合模型和 GDN 混合模型。两种混合模型均在每四层中使用一个全注意力层；其余的 token 混合层分别使用 SWA 或 GDN，其中 SWA 层的窗口大小为128。每个检查点对应一个基于 Qwen3.5 架构的 28 层 25B-A3B MoE 模型，该模型首先在 400B 个 token 上进行预训练，上下文长度为 4K，随后在 800B 个 token 上进行预训练，上下文长度为 32K。我们报告了在MMLU、MMLU-Pro 和 SuperGPQA 上的知识测试结果；在 MATH 和 GSM8K 上 STEM 测试结果；在 BBH 上推理测试结果；在 MMMLU 上多语言测试结果；以及在 EvalPlus 和 MultiPL-E 上代码测试结果。结果如表1所示。

GDN 混合模型在九项选定的基准测试中，有八项优于 Transformer 模型，七项优于 SWA 混合模型。它在七项基准测试中取得了最佳成绩，并拥有最高的总体平均分；而 SWA 混合模型在 MMLU 测试中略胜一筹，在 EvalPlus 测试中表现更佳。

<img
  src="https://i-blog.csdnimg.cn/direct/e899a42891db410f970295f40a7ecd5c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

#### 2.1.2 Qwen Sparse Attention

<img
  src="https://i-blog.csdnimg.cn/direct/f1f99b14dd4941249d1601aabb9f8c5c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Motivation**。通过选择性地关注重要上下文，稀疏注意力机制缓解了 softmax 注意力机制在长上下文场景下的二次方计算瓶颈。最近，DSA 利用轻量级索引器生成 token 级稀疏掩码，显著提升了推理速度。然而，随着序列长度的增加，其 $O(n^2)$ 索引器的开销仍然不可忽略。

为了降低索引开销，Qwen3.8-Flash-Next 采用了 **Qwen Sparse Attention (QSA)**。具体来说，QSA 使用轻量级索引器将序列压缩成微块表示，估计其重要性，并选择最相关的上下文进行注意力计算。这种设计在平衡任务性能和推理效率的同时，降低了长输入的索引开销。与跨层共享索引的方法相比，QSA 中的层内序列压缩较少依赖于跨层相似性，因此天然适用于混合架构。

图 3 展示了 QSA 架构。首先，一个压缩的轻量级索引器估计微块粒度的上下文重要性，然后指导核心注意力模块中的稀疏计算。

**Compressed Lightweight Indexer**。给定一个输入隐藏状态 $\textbf x_i$，索引器采用具有 $H$ 个 query head 和一个共享 key head 的 MQA。它首先应用独立的轻量级投影：

```math
\hat{\textbf q}^h_t=RMSNorm(\textbf W^h_Q\textbf x_i),\quad\textbf k_i=\textbf W_k\textbf x_i.\tag{12}
```

为了缩短序列长度，key 被分割成互不重叠的 $r$ 个 token 的块，并通过平均池化进行压缩。用 $p_b = b · r$ 表示块 $b$ 的起始位置，则对应的压缩 key 为

```math
\hat{\textbf h}_b=RMSNorm(AvgPool(\textbf k_{p_b:p_b+r-1})),\quad 0\le b\lt\lfloor\frac{n}{r}\rfloor.\tag{13}
```

对于位置编码，索引器应用部分 RoPE。具体来说，每个索引头的 128 个维度中有 64 个维度应用部分 RoPE，与核心注意力模块中使用的旋转维度相匹配。如公式（13）所示，key 压缩在位置编码之前执行。因此，每个块首先被概括为内容表示，然后被分配一个块级位置。这种顺序避免了对具有不同旋转相位的 token 表示进行平均。每个 query 保留其 token 位置 $i$，而每个压缩 key 被分配其所在块的起始位置 $p_b$：

```math
\textbf q^h_i=PRoPE(\hat{\textbf q}^h_i,i),\quad\bar{\textbf k}_b=PRoPE(\hat{\textbf k}_b,p_b).\tag{14}
```

然后通过块因果评分获得块级重要性得分 $\textbf I$。对于 query token $i$ 和压缩块 $b$，所有索引器 head 通过 ReLU 激活 query-key 相似度结果进行相加：

```math
I_{ib}=\begin{cases}
\sum^H_{h=1}ReLU\left(\langle\textbf q^h_i,\bar{\textbf k}_b\rangle\right), & p_b+r-1\le i,\\
−∞, & otherwise,
\end{cases}\tag{15}
```

这种块因果条件使得每个 query 仅对已完全观测到的块进行评分。给定一个 token 预算 $K$，每个 query 选择得分最高的压缩块。由于每个块包含 $r$ 个 token，因此块预算为 $K_B = ⌈K/r⌉$：

```math
\mathcal B_i=TopK_{K_B}(\{I_{ib}\}_b),\quad K_B=⌈\frac{K}{r}⌉.\tag{16}
```

然后，将选定的块扩展为它们的原始 token 索引，并截断为预算 $K$。**它们与始终包含在最终不完整块中的 token 一起，构成了用于核心注意力计算的最终集合**。

**Training Details**。QSA 是在 Qwen3.8-Flash-Next 的持续预训练（CPT）阶段引入的，序列长度为 256K tokens。训练过程包括两个阶段：密集蒸馏和稀疏训练。
- **Stage 1: Dense Distillation**。我们首先将骨干网络的全序列注意力分布提炼到索引器中。通过对所有教师 head 的 softmax 注意力分布求和并应用 L1 归一化，得到 token 级教师分布。将从 query $i$ 到 token $j$ 的概率记为 $a_{ij}$，则完整的 token 级分布为 $a_i∈\mathbb R^n$。沿用先前的工作，我们应用最大池化来使该分布与块级索引器得分对齐，从而保留在聚合过程中可能被稀释的显著 token 级信号：
```math
\bar a_{ib}=MaxPool(\textbf a_{i,p_b:p_b+r-1}),\quad\hat{\textbf a}_i=\frac{\bar{\textbf a}_i}{||\bar{\textbf a}_i||_1}.\tag{17}
```
令 $B = ⌊n/r⌋$，则合并后的教师分布 $\hat{\textbf a}_i ∈ \mathbb R^B$ 和指标得分 $\textbf I_i ∈ \mathbb R^B$ 具有相同的块维度。通过最小化以下目标函数，将归一化的教师得分蒸馏到指标中：
```math
\mathcal L_{KL}=\frac{1}{N}\sum_iD_{KL}(\hat{\textbf a}_{i,:}||Softmax(\textbf I_{i,:})),\tag{18}
```
其中 $N$ 为 query token 数。每个 query 的 KL 损失仅包含完整的 key 块。在预热阶段，仅对索引器进行 1000 步训练，学习率为 $1 × 10^{-3}$。每步包含 8 个 256K 个 token 的序列，总共约 2B 训练 token。
- **Stage 2: Sparse Training**。索引器初始化后，在索引器的指导下训练整个骨干网络，以适应稀疏注意力模式。根据公式（16），QSA 首先选择 top-$K_B$ 个块。这些块被扩展为 token 索引，并与尾部 token 组合成最终的不完整块：
```math
\mathcal S_i=Expand(\mathcal B_i) ∪\left\{r\lfloor\frac{i+1}{r}\rfloor,...,i\right\}.\tag{19}
```
$Expand$ 算子将选定的块映射到它们的 token 索引。得到的集合 $\mathcal S_i$ 用于核心注意力计算。在此阶段，索引器 $KL$ 损失仅针对 $B_i$ 中的 top-$K_B$ 个块进行计算。在评估 KL 散度之前，$\mathcal B_i$ 中的教师概率被重新归一化，使其总和为 1：
```math
\mathcal L_{KL}=\frac{1}{N}\sum_iD_{KL}(\hat a_{i,\mathcal B_i}||Softmax(\textbf I_{i,\mathcal B_i})).\tag{20}
```
在 CPT 的最后阶段，启用 QSA，并使用 $2.5 × 10^{−5}$ 的学习率对骨干网和索引器进行 8,000 步联合训练。每一步包含 96 个 256K 个 token 的序列，总共约 200B 个训练 token。

**Implementation Details**。对于 Qwen3.8-Flash-Next，骨干网络和 MTP 模块中的所有全注意力层都被替换为 QSA，以提高长序列的推理效率。压缩后的轻量级索引器采用 MQA 结构，包含四个 query head 和一个共享 key head，并使用上述的部分 RoPE 配置。在 token 预算 K = 2048 和压缩比 $r = 4$ 的情况下，QSA 为每个 query 选择最多 512 个完整块，并在最后一个不完整块中包含尾部 token。索引器提供的精确重要性估计使得模型在第二阶段稀疏训练期间能够紧密匹配全注意力的 LM 损失轨迹。如图 4 所示，两条损失曲线保持高度一致，总体损失差异约为 $10^{-4}$。

为了高效训练，我们实现了一种融合的 QSA 核，它联合计算稀疏注意力输出和 KL 损失，无需生成中间结果，从而显著降低了内存消耗。压缩后的索引器对序列进行压缩后的评分，减少了索引器计算和 top-k 个结果选择的开销。此外，多步 MTP 在预测步骤中复用 top-k 个结果索引，进一步降低了草稿模型的推理成本。

<img
  src="https://i-blog.csdnimg.cn/direct/a9fb08c4f9254244a53057f36ec5badf.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Evaluation of QSA**。我们通过将配备稀疏注意力机制的 Qwen3.8-Flash-Next 模型与其完全注意力基线模型进行比较来评估 QSA。表 2 报告了知识、STEM、推理、多语言和编码基准测试的结果，从而全面评估了稀疏注意力机制是否会影响模型在长上下文检索之外的通用能力。在八个基准测试中的七个测试中，QSA 的性能与完全注意力基线模型持平或更优，并将平均得分从 75.9 提高到 76.8。其余差异很小，并未表明在任何任务类别中存在系统性性能下降。总体而言，这些结果表明，QSA 在保持模型在短上下文任务上的通用能力的同时，能够更高效地进行长上下文推理。

除了短上下文基准测试之外，我们还评估了 QSA 在两个广泛使用的长上下文基准测试 RULER 和 MRCR 上的检索能力。我们评估了 RULER 在 4K 到 1000K 序列长度下的性能，并使用 8 针 MRCR 设置评估了 128K 到 1M 序列长度下的性能，如表 3 所示。在较短的序列长度下，QSA 的性能与完全注意力机制相当，并且随着上下文长度的增加，性能更佳。特别是，QSA 将 RULER 得分从 90.08 提高到 93.00（超过 512K）。在 MRCR 上，得分在 512K 时从 30.66 提高到 40.53，在 1M 时从 20.71 提高到 26.44。这些结果表明，QSA 能够在不牺牲任务性能的前提下提高长上下文推理效率，并在更长的序列长度上带来进一步的性能提升。

<img
  src="https://i-blog.csdnimg.cn/direct/310868f8bfd64d689c1d9077970489ca.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们进一步研究了 QSA 对多 token 预测（MTP）模块的影响。除了用 QSA 替换 MTP 中的注意力层之外，我们还遵循 GLM，并在推测解码步骤中复用前 k 个索引，以进一步提高草稿模型的效率。为了评估复用 QSA 是否会影响 MTP 的性能，我们在来自不同领域的基准数据集上进行了四步推测解码实验，结果如表 4 所示。结果表明，复用 QSA 后，平均接受长度没有显著变化。

<img
  src="https://i-blog.csdnimg.cn/direct/87fcb24d6c4e4d4496799f8d595adc7f.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Architecture Ablation**。我们对 35B-A3B 规模的 QSA 进行架构消融，重点关注压缩比和索引头数量。如图 5 所示，我们使用序列长度高达 1M 的 RULER 分数作为评估指标。所有实验均使用经过第二阶段稀疏训练后获得的 CPT 模型进行评估。

为了**评估压缩比**，我们使用不同的微块大小来评估 QSA。此外，我们还引入了训练感知型 IndexShare 作为基线，它通过在混合模型中相邻的全注意力层之间均匀共享前 K 个索引来降低索引开销。在图 5(a) 中，我们展示了两种方法的 RULER 性能与其估计的索引器延迟降低值之间的关系。QSA 在相对索引器延迟为 0.25 时与全注意力基线性能相当，而 IndexShare 在 0.5 时仍低于基线。对于 IndexShare，0.5 表示在两个由三个 GDN 层隔开的全注意力层之间共享单个索引。该结果突显了混合架构中层内压缩的优势，因为在混合架构中，跨层索引共享可能受到层间相似度低的限制。

query head 的数量直接影响索引器的效率，尤其是在预填充阶段。因此，我们在 RULER 数据集上评估了不同 query head 数量的 MQA 索引器，如图 5(b) 所示。密集初始化后，直接应用稀疏注意力索引器会导致性能明显下降。短暂的联合训练可以让骨干网络适应稀疏注意力模式，并恢复到全注意力水平。结果表明，QSA 仅需少量索引器 query head 即可保持性能，远少于核心注意力模块所使用的查询头数量。为了平衡推理速度和准确性，我们最终采用 4 个 query head 作为轻量级索引器配置。

<img
  src="https://i-blog.csdnimg.cn/direct/0cae2a597dd54ee8ba514bb83cdbff63.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Efficiency Analysis**。通过以压缩比 $r$ 对 key 序列进行压缩，QSA 将索引器的复杂度从 $O(n^2)$ 降低到 $O(n^2/r)$，从而在处理长序列时显著提升效率。对于索引器而言，序列压缩直接减少了 MQA logits 和 top-k 选择的计算量，从而实现了与压缩比相当的加速。如图 6(a,b) 所示，这显著降低了长序列的推理成本，而索引器在长序列上往往成为主要的瓶颈。

我们进一步评估了 QSA 的内核级加速，同时考虑了索引器开销和稀疏核心注意力计算。对于密集注意力基线，我们使用了 FlashInfer 提供的分页 GQA 实现。预填充在分块预填充设置下进行评估，而解码则包含三个额外的 MTP 步骤。如图 6(c,d) 所示，QSA 从上下文长度 64K 开始即可提供加速，并且随着序列长度的增加，加速效果越来越显著。在上下文长度为 1M 时，QSA 在预填充和解码过程中分别实现了 7.6 倍和 4.9 倍的注意力模块加速，证明了其在长上下文推理中的可扩展性。

### 2.2 Residual

**Motivation**。残差连接为每个模块提供了一条直达网络输出的路径。Pre-normalization 能够保证大规模训练的稳定性，但它会削弱每一层接收到的信号：每个模块读取的是相同的数据流，因此早期写入的特征必须与之后写入的所有特征竞争。一些研究工作通过添加绕过瓶颈的路径来解决这个问题，包括密集的层间连接、用于注意力值的额外残差路径以及跨层重用缓存状态。

直接修改残差路径的工作可分为两类。第一类使每一层的读写操作更具表现力，类似于 highway 网络。第二类则拓宽了数据流本身： Alternating Updates (AltUp) 和 Hyper-Connections (HC) 用多个并行分支取代了单个残差向量。这两类工作相辅相成：拓宽数据流增加了容量，而更丰富的读写机制则决定了如何使用这些容量。

**Widening the Residual Stream**。我们首先研究报告的增益有多少仅来自宽度，为此我们使用了一个简化的 AltUp 变体，该变体适用于 pre-norm 网络。块 $ℓ$ 之前的残差状态是 $n_r$ 个分支的集合 $\textbf R^{(ℓ)} ∈ \mathbb R^{n_r×d}$，其中 $d$ 是隐藏层的大小，$\textbf R^{(ℓ)}_i$ 表示分支 $i$。每个块包含 $n_r$ 个可学习标量 $\textbf h ∈ \mathbb R^{n_r}$，其输入读取为分支的加权和。

```math
\textbf x^{(ℓ)}=\sum^{n_r}_{i=1}h_i\textbf R^{(ℓ)}_i.\tag{21}
```

然后，将块输出 $\textbf y^{(ℓ)}$ 写回单个分支，该分支按深度轮询选择：

```math
\textbf R^{ℓ+1}_{i}=\textbf R^{(ℓ)}_i+\textbf 1[i=ℓ~mod~n_r]\textbf y^{(ℓ)}.\tag{22}
```

这会在每个块中增加 $n_r$ 个参数，且无需矩阵乘法，因此其计算成本可以忽略不计；额外的成本在于需要传输 $n_r$ 个分支而不是一个分支，从而增加了内存流量。即便如此，它也能将基于 400B token 训练的 25B-A3B MoE 模型的训练损失降低约 0.01。因此，仅加宽度本身就能显著降低损失。剩下的问题是，为了在性能、效率和稳定性之间取得良好的平衡，应该在加宽的流之上添加多少读/写机制。

**Hyper-Connections (HC)**。HC 将公式 (21) 和公式 (22) 推广为三个可学习的算子。读取算子 $\textbf H_{mix}$ 形成块输入，写入算子 $\textbf H_{combine}$ 将块输出分配到各个分支，混合算子 $\textbf H_{res}$ 在分支之间交换信息：

```math
\textbf x^{(ℓ)}=\textbf H^T_{mix}\textbf R^{(ℓ)},\tag{23}
```

```math
\textbf y^{(ℓ)}=\mathcal F^{(ℓ)}\left(Norm(\textbf x^{(ℓ)})\right),\tag{24}
```

```math
\textbf R^{(ℓ+1)}=\textbf H_{res}\textbf R^{(ℓ)}+\textbf H_{combine}\textbf y^{(ℓ)T},\tag{25}
```

其中 $\textbf H_{mix}, \textbf H_{combine} ∈ \mathbb R^{n_r}$ 且 $\textbf H_{res} ∈ \mathbb R^{n_r×n_r}$。在 HC 符号表示中，它们分别表示为 $\textbf A_m, \textbf B$ 和 $\textbf A_r$。这三个算符由残差态预测得出。令 $\bar{\textbf R} = norm(\textbf R^{(ℓ)})$ 表示分支的归一化表示。每个算符都是一个静态项和一个数据相关项之和：

```math
\textbf H_{mix}=\textbf H^{s}_{mix}+\textbf λ_m\odot ϕ(\bar{\textbf R}\textbf W_m),\tag{26}
```

```math
\textbf H_{combine}=\textbf H^s_{combine}+\textbf λ_c\odot ϕ(\bar{\textbf R}\textbf W_c),\tag{27}
```

```math
\textbf H_{res}=\textbf H^s_{res}+\textbf λ_r\odot ϕ(\bar{\textbf R}\textbf W_r),\tag{28}
```

其中 $\textbf H^s_⋆$ 是一个可学习的静态项，$\textbf W_⋆$ 投影归一化残差，$ϕ$ 是一个激活函数，$\textbf λ_⋆$ 是一个可学习的标量，$⋆$ 代表三个算子中的任意一个。HC 使用 $ϕ = tanh$，$\textbf λ_⋆$ 初始化为 $0.01$；mHC 使用 sigmoid 函数，并且额外将 $\textbf H_{res}$ 约束为双随机矩阵流形。将静态项设置为 $e_{ℓ~mod~n_r},\textbf 1$ 和 $\textbf I$，且 $\textbf W_⋆ = 0$，则扩展网络恰好从 pre-norm 网络开始；当 $\textbf λ_⋆ = 0$ 时，算子保持静态，扩展流无需额外计算，从而恢复了上述简化的 AltUp 变体。

**Design Ablation**。从静态运算符出发，我们仅在需要增加表达能力时才保留这些能力。表 5 列出了这一改进过程的最终结果，这些结果使用 §2.1.1 中的基准测试套件和评估流程进行了评估。最终设计基于以下五点观察结果。
- **Bounded positive gates**。在损失函数和训练稳定性方面，sigmoid 门控函数都优于 tanh 门控函数。这与 mHC 的结果一致，也与 GDN 和注意力组件中观察到的 sigmoid 门控函数优于 SiLU 或 tanh 的现象相符。
- **Data dependence**。将 $\textbf H_{mix}$ 和 $\textbf H_{combine}$ 算法与数据关联，相比静态版本，损失降低了 0.002；而静态版本相比 pre-norm 基线，损失降低了 0.021。基准测试的增益则相反：从静态到动态提升了 1.98 分，而从基线到静态则提升了 1.58 分（表 5）。这是本报告中损失和下游准确率变化不一致的几个例子之一。
- **Read granularity matters more than write granularity**。将公式 (23) 中的 $\textbf H_{mix}$ 从每个分支一个标量细化为每个分支和通道一个权重，效果显著。对公式 (25) 中的 $\textbf H_{combine}$ 进行同样的细化几乎没有任何效果，因此写入操作仍然是每个分支一个标量。
- **Read all branches**。预测所有分支的运算符比仅使用最后一个分支或先将所有分支合并要好。分别对每个分支进行归一化，即对加宽后的数据流进行分组 RMSNorm 运算，可以进一步提高性能。
- **$\textbf H_{res}$ adds little**。一旦读取和写入操作足够强大，添加 $n_r × n_r$ 混合运算符就不会带来显著的改进。

使用静态运算符拓宽数据流已经比未归一化的基线平均准确率提高了 1.58 个百分点，而使读写操作与数据相关则进一步提高了 1.98 个百分点。mHC 静态和动态之间的性能损失差距仅为 0.002，但基准测试结果的差距却相当显著；在这种情况下，仅考虑性能损失会低估改进的价值，这也强化了我们在整个设计过程中同时检查性能损失和基准测试结果的做法。下文将详细介绍 GR 的具体设计。

<img
  src="https://i-blog.csdnimg.cn/direct/983e924c0a9c4e78bc201307f0a94e40.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Gated Residual**。在另一项研究中，我们发现，在 RMSNorm 之后添加一个轻量级的逐元素自门控（我们称之为 GatedNorm）可以显著提高训练稳定性：

```math
GatedNorm(\textbf u)=RMSNorm(\textbf u)\odot\sigma(\textbf W_2SiLU(\textbf W_1RMSNorm(\textbf u))),\tag{29}
```

其中 $\textbf W_1$ 和 $\textbf W_2$ 构成低秩瓶颈。消融后得到的读取结果，是逐元素且数据相关的，并带有 sigmoid 门控，它恰好是公式 (29) 应用于加宽后的流。因此，我们将两者合并为一个算子，并将结果称为 Gated Residual (GR)。

GR 首先对每个分支进行独立归一化，

```math
\hat{\textbf R}_i=RMSNorm(\textbf R_i, γ_i),\quad i=1,...,n_r,\tag{30}
```

每个分支都有自己的增益 $γ_i ∈ \mathbb R^d$。然后，它根据所有分支预测每个分支和通道的逐元素门控分数，并将门控分支平均到模块输入中：

```math
\textbf G=unvec~\sigma\left(\textbf W_uSiLU(\frac{1}{n_r}\textbf W_dvec(\hat{\textbf R}))\right),\tag{31}
```

```math
\textbf x=\frac{1}{n_r}\sum^{n_r}_{i=1}\textbf G_i\odot\hat{\textbf R}_i,\tag{32}
```

其中 $vec$ 将分支堆叠成一个长度为 $n_rd$ 的向量，unvec 是它的逆向量，对于瓶颈秩 $r = d/8$，$\textbf W_d ∈ \mathbb R^{r×n_rd}$ 且 $\textbf W_u ∈ \mathbb R^{n_rd×r}$。块输出 $\textbf y = F(\textbf x)$ 通过每个分支一个数据相关的标量写入每个分支：

```math
\textbf s=2σ\left(\frac{1}{n_r}\textbf W_wvec(\hat{\textbf R})\right )\in\mathbb R^{n_r},\tag{33}
```

```math
\textbf R'_i=\textbf R_i+s_i\textbf y,\tag{34}
```

其中 $\textbf W_w ∈ \mathbb R^{n_r×n_rd}$。式 (31) 和式 (33) 分别是式 (26) 和式 (27) 的读写算子，$ϕ = σ$，分别表示逐元素 $\textbf H_{mix}$，每个分支的标量 $\textbf H_{combine}$，所有分支的组 RMSNorm $\bar R$。静态项 $\textbf H^s_⋆$ 对 GR 没有改进。在当前配置下，无需对可学习权重进行特殊初始化，静态项的贡献可以忽略不计；使用骨干网络中使用的标准随机初始化即可。我们使用 $n_r = 4$ 个分支，每个层的注意力模块和 MLP 模块分别使用独立的 GR 模块。

在表 5 中，GR 和 mHC（动态）的主要区别在于逐元素混合和移除 $\textbf H_{res}$；在此规模下，两者的性能相当。GR 的效率优势在于移除 $\textbf H_{res}$ 消除了每个块对残余状态的完整读取，从而减少了内存流量。稳定性优势体现在两个方面：GatedNorm 本身提高了训练稳定性（详见 §3.3），而移除 $\textbf H_{res}$（需要单独的约束）则消除了一个潜在的不稳定源。

GR 与 HC、mHC 和 VWN 属于同一类算法；区别在于它们如何分配额外的表达能力。HC 和 mHC 将读取和写入操作保留为每个分支的标量，并将容量投入到 $\textbf H_{res}$ 中，mHC 还进一步限制 $\textbf H_{res}$ 为双重随机状态。VWN 也保留了这些标量，但它拓宽了 token 嵌入，将其分割成许多细小的片段；这种分割同样追求更细粒度的读取和写入操作。GR 将其表达能力用于读取操作：门操作是逐元素进行的，并且完全舍弃了 $\textbf H_{res}$，消融实验表明这不会增加任何成本，并且消除了每个块对整个残余状态的读取操作，而这正是拓宽流的主要推理成本。

由于公式 (32) 中的读取操作已经对与门进行了归一化，GR 操作也取代了块的 pre-norm，而不是置于其前面：公式 (24) 失去了 Norm，并且扩展操作没有增加任何归一化层。由于没有混合算子，分支保持分离，因为分支只能由块写入，并通过公式 (32) 读取。这使得信息流易于追踪，我们在 §2.2 的分析中使用了这一点。

**Comparison with Attention Residual**。残差注意力（AttnRes）利用 softmax 注意力机制对前几层的输出进行注意力控制，从而确定每个子层的读取信息。完整残差注意力（Full AttnRes）则关注每个前一层子层的输出。块级残差注意力（Block AttnRes）将 $L$ 个子层划分成若干个大小为 $S$ 的块，将每个块相加得到一个表示，然后关注这个表示。

表 6 在我们的设置（一个 28 层模型，$L = 56$ 个子层）中，将两种变体与 GR 进行了比较，分别在有无 GN 的情况下进行了比较。Full AttnRes 是该系列中最强的设置，其损失与 GR 相当，为 1.762；而汇总块在 $S = 2$ 时损失为 0.008，在 $S = 4$ 时损失为 0.011。同样的排序在更深的层数下仍然成立：在 48 层时，$S = 4$ 的 Block AttnRes 的损失达到 1.711，而 GR 的损失为 1.707。GN 在所有设置下都降低了损失，在 AttnRes 上降低了 0.004–0.005，在普通预范数基线上降低了 0.002。当子层读取的输入更复杂时，门的作用更大，这与 GR 在其读取操作中包含的门相同。

<img
  src="https://i-blog.csdnimg.cn/direct/8e10dafb0bf14838b9323111fb0703d6.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**What the Branches Are Used For**。消融实验表明，GR 改进了损失函数和基准测试，但其增益机制是什么？由于 GR 没有残差分支混合，每个分支都只是过去输出的简单累加器，我们可以将每个块的读取内容精确分解为来自之前所有块的贡献。我们利用这种分解方法，将 GR 模型与一个其他方面完全相同的未采用 GR 的参考模型进行比较，发现额外的宽度用于几个特定的​​路径：**一个分支保留了跨多个层的早期注意力输出，而其他三个分支则保持局部性**。

*The statistic*。在块 $v$ 之前，分支 $c$ 保存的是

```math
R_c^{(v)} = R_c^{(0)} + \sum_{u<v} s_c^{(u)}\, y^{(u)},
\tag{35}
```

其中，$R_c^{(0)}$ 是分支 $c$ 的初始值（即 token embedding），$y^{(u)}$ 是块 $u$ 的输出，而 $s_c^{(u)}$ 是式（33）中的标量写入门控，用于控制 $y^{(u)}$ 中有多少进入分支 $c$。

块 $v$ 通过式（32）中的门控读取来读取该分支：它对该分支进行归一化，使用学习得到的逐通道增益 $\gamma_c$（来自式（30））对其进行缩放，再使用 $G_c^{(v)}$（式（31）中的数据依赖门控）对结果逐元素进行门控，并在各分支之间取平均。由于归一化会除以 $\operatorname{rms}\!\left(R_c^{(v)}\right)$，因此块 $u$ 对块 $v$ 输入的贡献为

```math
a_{u\to v}=
\frac{1}{n_r}
\sum_{c=1}^{n_r}
G_c^{(v)}
\odot
\gamma_c
\odot
\frac{s_c^{(u)}\,y^{(u)}}{\operatorname{rms}\!\left(R_c^{(v)}\right)},
\tag{36}
```

其中使用的是前向传播实际采用的门控值。每个因子都有直接的含义：$s_c^{(u)}\,y^{(u)}$ 是块 $u$ 写入分支 $c$ 的内容；除以 $\operatorname{rms}\!\left(R_c^{(v)}\right)$ 是为了计入读取时应用的归一化；$\gamma_c$ 对每个通道进行重新缩放；而 $G_c^{(v)}$ 决定块 $v$ 实际使用该分支中的多少内容。

我们报告这一贡献的归一化幅度，

```math
\pi_{uv}=
\frac{\left\|a_{u\to v}\right\|}
{\sum_{u'<v}\left\|a_{u'\to v}\right\|},
\tag{37}
```

即块 $v$ 的输入中来自块 $u$ 的比例。由于式（36）在求和之前按分支拆分每一个贡献，因此我们可以看出每条路径由哪个分支承载。该分解是精确的：对于每个读取者，各个份额之和与 1 的误差不超过 $3\times10^{-8}$。

我们将一个使用 GR 训练的 20 层 MoE 与一个除此之外完全相同、但不使用 GR 的参考模型进行比较；二者使用相同的训练配置、数据、优化器和训练步数，并在相同的 token 上进行探测。我们报告差值

```math
\Delta_{uv}=
\pi_{uv}^{\mathrm{GR}}-
\pi_{uv}^{\mathrm{ref}},
```

因为减去参考模型可以去除所有残差网络共有的模式，即邻近的写入者占主导。

*What the figure shows*。图 7 展示了 21 条路径（在 780 个有序对中），这些路径至少跨越一层，并满足 $\Delta_{uv}\ge 0.05$。作为尺度参考：在第 15 层有 30 个写入者，因此如果平均分配，每个写入者的份额约为 0.03；而 0.13 的份额大约是这一数值的四倍。

一个分支承载长程路径，而另外三个分支保持局部。具体是哪一个分支并不重要，因为在初始化时这些分支是可互换的，但这一模式是一致的：在五个 GR checkpoint 中，每一个都恰好有一个这样的分支，其典型跨越距离为 10.9 层，而其余分支为 3.4–3.9 层。下面三个例子说明了这一模式；每个例子都先给出参考模型（不使用 GR 的单一残差流）下的份额，然后给出 GR 下的份额。

* 第 0 层 GDN $\rightarrow$ 第 15 层 attention：该份额从参考模型中的 0.020 上升到 GR 中的 0.138。从第 10 层到第 19 层的每一个读取者，这一份额都保持在 0.072–0.138，并且没有下降趋势。这证实了第一层在跨深度保留信息方面具有超乎寻常的重要作用，与先前的观察一致。

* 第 10 层 GDN $\rightarrow$ 第 11 层 attention：$\Delta_{uv}=0.117$。这一增益与上面的路径一样大，但只跨越单层：GR 同样增强了短程连接。

* 第 0 层 MLP，同时作用于两个分支：在长程分支上通向第 15 层，其份额从参考的无 GR 模型中的 0.008 上升到 GR 中的 0.058；同时，在一个局部分支上通向第 2 层，其份额从 0.139 上升到 0.192。同一个输出以不同的强度到达一个邻近读取者和一个远距离读取者，而单一流无法表达这一点，因为对于每个写入者，它只有一个衰减速率。

分支 0 中典型的长程路径可以由式（35）得到解释：该分支在第 0 层被最强烈地写入，此后几乎不再更新，因此第 0 层的信息对之后的所有层都保持可访问。此外，从 GR 分支中读取最强的子层主要是 softmax attention 层，这表明 global attention 充当了一个关键枢纽，用于整合 GDN 层所压缩掉的显式长程历史上下文。

为了观察整体模式，我们按照路径跨越的层数对全部 780 条路径进行分组，并在每个组内对 $\Delta_{uv}$ 求和，其中 $\Delta_{uv}$ 表示 GR 相对于参考模型为路径 $(u,v)$ 额外赋予的份额。结果是：相邻层路径（skip 1）的总份额增加 0.96；长程路径（skip $>12$）的总份额增加 0.91；而中程路径（skip 2–12）的总份额减少 3.21。加权平均的跨越距离几乎没有变化（3.97 对 3.91），因此跨层信息的总量是相近的；发生变化的是它的分布。GR 会选择少数几条路径并将其放大，而代价是削弱中程路径。


<img
  src="https://i-blog.csdnimg.cn/direct/44f111c645d24ee2a7856dd31c424454.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Inference Efficiency**。GR 的推理成本主要由扩宽后的残差状态所产生的内存流量主导。因此，我们寻找了减少数据搬运字节数的方法。

第一次尝试是将读取稀疏化。我们观察到，在训练完成的模型中，每个 GR 层的写入通常由两个分支主导。因此，我们尝试引入稀疏写入，无论是从训练开始就引入，还是在训练中途引入，其中每个块只读取门控值最高的两个分支，而不是读取全部 $n_r$ 个分支。预训练损失和基准测试几乎没有受到影响，但在后训练之后，模型质量出现了明显下降，因此我们没有采用这种方法。更复杂的变体，例如在不同层之间改变稀疏程度，也没有解决这一问题。我们指出，*这是一个仅依靠预训练指标会导致错误决策的案例。* 最近的工作 xHC 探索了使用更大的 $n_r$ 来使稀疏分支更新更加容易，但考虑到更大的 $n_r 会带来额外的内存开销，我们没有进一步探索这一方向。

第二次尝试是将残差状态保持为 FP8。GR、gated attention 和 GDN 中的门控都会限制写入流中的数值幅度，因此残差值保持在一个较窄的范围内，非常适合低精度格式。与 BF16 相比，以 FP8 存储这些分支可以将残差状态的数据搬运字节数减半，同时几乎不会造成质量损失。最后，式（30）–（32）的读取操作和式（33）–（34）的写入操作分别融合到单个 kernel 中，并将 group RMSNorm 融合到读取操作中，因此对于每个块，扩宽后的流在每个方向上都只需遍历一次。

### 2.3 N-gram Embedding

**Motivation**。基于嵌入的 memory 为扩展模型容量提供了一个补充维度。N-gram 嵌入通过基于局部上下文而非仅基于 token 标识来约束 memory 检索，从而进一步泛化了单 token 查找。具体而言，以每个 token 结尾的短 n-gram 作为嵌入表的键，检索到的向量则增强了相应的 token 表示。N-gram memory 以几乎不增加每个 token 的浮点运算次数来扩展容量，同时确定性寻址实现了主机内存卸载和异步预取。在本节中，我们将系统地研究 N-gram 嵌入的关键架构选择。我们使用多头哈希进行嵌入查找。嵌入通过上下文门控机制进一步注入到残差流中。在所有实验中，训练数据量统一按“每个实际参与计算的参数，分配 300 个训练 token”来设定。

#### 2.3.1 Placement of the N-gram Embedding Layers

<img
  src="https://i-blog.csdnimg.cn/direct/2a744d4875684dad84234f7b266eb30c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们在固定的参数预算下，调整了 N-gram 嵌入层的位置和数量。单层变体涵盖了浅层（第1-4层）、中层（第10和15层）和深层（第25层）三种位置。对于多层配置，我们将一个浅层（第2层）与一个中层（第15层）或一个深层（第25层）相结合。结果如表7所示。

没有哪一种深度方案始终占据主导地位。前两层表现优异，而中间层和深层也保持竞争力。将相同的参数预算分配到多层并不能带来持续的收益。合并第 2 层和第 25 层带来的边际损失降低并不能转化为下游性能的提升。单个 N-gram 嵌入层就足够了。此外，在完全注意力机制和 GDN（§2.1.1）下，不同位置的相对性能相似，这表明位置选择对注意力机制基本不敏感。**我们将 N-gram embedding 放在第 2 层，这样 host-memory 预取就可以与第一层的计算重叠**。

#### 2.3.2 N-gram Vocabulary Size

我们进一步探讨了扩展 N- gram 嵌入的词表大小的影响。

<img
  src="https://i-blog.csdnimg.cn/direct/26d77dc7232645899427456a0cc811f8.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Allocation under Fixed Parameter Budget**。沿用前人的工作，我们在保持总参数预算不变的情况下，通过减少专家数量来扩展 N-gram 嵌入槽位。结果如表 8 所示。损失随词表大小的变化呈非单调性，在词表大小为 10 倍时最低（25%），这与前人工作中报告的分配最佳点一致。在其他评估中，并未发现相同的最优值。域外不可作弊的 PPL 在不同预算下变化不大，下游基准测试也未显示出相对于仅使用 MoE 的基线有明显的改进。这些结果表明，N-gram 嵌入和 MoE 专家在容量扩展方面发挥着不同的作用。因此，我们研究在保持 MoE 参数预算不变的情况下，如何扩展 N-gram 词表大小。

<img
  src="https://i-blog.csdnimg.cn/direct/1bf56000ff5642d8b74f5a2cd30ebad3.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Scaling with Additional Parameters**。由于嵌入表的访问稀疏且寻址确定，因此可以以极小的额外计算量进行扩展，并存储在加速器外的存储空间中。因此，我们突破了固定大小的限制，将 N-gram 词表从 20V 扩展到 200V，其中 V 表示 Qwen3.5 的基础 tokenizer 词表大小。结果如表 9 所示。损失值随着 N-gram 词表的增长而单调递减，但下游性能并未呈现相同的趋势。词表扩展相比基线方法带来了广泛的性能提升，但在某些基准测试中，随着词表的增长，性能会出现饱和或波动。此外，在中文基准测试（例如 C-Eval 和 CMMLU）上的性能也随着 N-gram 词表大小的增加而持续提升。

除了词表扩展之外，我们还探索了一系列提高 N-gram 嵌入参数效率的策略，包括但不限于用于词表压缩的 token 归一化、跨 N-gram 顺序的非均匀分配以及基于词频的嵌入槽划分。尽管做出了这些努力，但我们的训练方案并未取得持续的性能提升。

## 3.Optimization

### 3.1 Optimizer

**Motivation**。基于矩阵的优化器 Muon 通过牛顿-舒尔茨 (NS) 迭代法对动量进行正交化，从而计算矩阵参数的更新方向，并且已被证明在大规模计算中有效。我们使用 Muon 作为主要优化器，并发现一些设计选择会影响其实际效率和稳定性，我们将在下文中进行描述。

**Orthogonalization**。NS 迭代应用于 Nesterov 加速动量（$µ = 0.95$），正交化结果乘以 $γ(A, B) = 0.2\sqrt{max(A, B)}$ 进行缩放，其中参数形状为 $A × B$，使得更新的均方根与矩阵形状无关。对于 NS 迭代，我们采用极地快车方法的逐步系数调度，该方法在给定步数预算下是极小极大最优的。我们将迭代步数设置为 8，这比更少的步数提供了更精确的正交化，并降低了压力测试中梯度范数尖峰的幅度和频率。我们将 NS 迭代之前的 Frobenius 归一化中的数值稳定性常数设置为 $10^{−14}$。

**Which Parameters Use Muon**。我们将 Muon 应用于那些**真正充当线性映射的二维权重**：注意力模块中的 $q/k/v$ 投影和输出投影、GDN 的输入与输出投影、路由专家（routed experts）和共享专家（shared experts）的 fc1/fc2，以及 N-gram embedding 层中的 key/value 投影。输入 embedding 和输出 head 仍然使用 AdamW。对于 MoE router，我们观察到 Muon 会加剧训练早期的波动，并使 router 变得不稳定。虽然在训练中后期对 router 使用 Muon 不会引发不稳定，但我们也没有发现显著的性能收益。因此，我们对 router 使用 AdamW。一种可能的解释是，router 的每个输出维度都对应某一个 expert 的分数，而这些维度在很大程度上彼此独立，因此不存在可供正交化利用的共享线性结构。GR（§2.2）的两个低秩投影同样在使用 AdamW 时表现更好。我们认为这是由于它们的形状非常狭长。最后，$n$-gram embedding table 使用 Adam，并关闭 weight decay。

**Splitting Fused Parameters**。在 Megatron-LM 中，attention 的 $qkv$ 投影、SwiGLU 的 fc1，以及 GDN 的输入投影，各自都被存储为一个融合矩阵（fused matrix）；但从语义上看，它们实际上是沿输出维度拼接起来的多个独立线性算子。因此，直接对融合矩阵执行正交化在两个方面都是错误的：一是迭代过程会在彼此无关的子块之间混合奇异方向；二是 $\gamma(A,B)$ 是根据拼接后矩阵的形状计算的，而不是根据真实线性算子的形状计算的。因此，我们会在正交化之前先拆分融合后的梯度，分别在每个子矩阵上独立运行 NS，然后在应用参数更新之前，将结果重新聚合回原始布局。$qkv$ 和 GDN 输入投影按照 **per-head 粒度**进行拆分，这可以同时改善 loss 和下游 benchmark。fc1 则拆分成 gate 和 up 两部分；loss 基本保持不变，而 benchmark 会略有提升。拆分还为将某些单独的子矩阵排除在 Muon 之外提供了自然的粒度。具体来说，GDN 的 decay 和 beta 投影会为每个 head 产生一个标量，因此它们实际上是向量，而对向量执行正交化并没有意义。对于输出 gate（attention output gate 以及 GDN 的 $z$ 投影），我们的消融实验发现 AdamW 与 Muon 表现相当，甚至略优于 Muon。

**Implementation**。NS 迭代给 Muon 带来了两个主要的实现挑战。第一，迭代要求对每一个完整参数矩阵进行整体更新（对于 $K$ 次迭代，其计算量约为 $\approx 4K\,\max(A,B)\min(A,B)^2\ \text{FLOPs}$），这与 Megatron 的分片机制存在两方面冲突：在 TP 下，没有任何一个 rank 持有完整的权重矩阵；而在 DP 下，由于计算成本相对于较短维度呈三次方增长，因此 Megatron 按元素数量均匀划分参数的方式会造成非常严重的 straggler。我们开发了 **Canzona**，将逻辑上的 optimizer 分配与物理参数布局解耦：一个 $\alpha$-balanced 静态分区器会重新分配完整参数（不会在 tensor 内部进行切分），以在不同 DP rank 之间均衡预计的 NS FLOPs；与此同时，一个异步的 **Micro-Group pipeline** 会通过 TP rank 之间融合后的 All-to-All，重新构建每一个由 Muon 负责的矩阵。随后，每个 owner 都会执行一个在数学上与单设备 Muon 等价的更新步骤；同时保留 ZeRO-1 的 bucket 几何结构，因此 Megatron 的 Reduce-Scatter / backward overlap 仍然能够得到保留，而且这一抽象还可以扩展到其他基于矩阵的 optimizer。第二，在完成拆分之后，单个 layer 中可能包含大约上百个子矩阵，optimizer step 因而会变成一长串非常小的 kernel；此时性能瓶颈不再是算术计算，而是 kernel launch overhead。我们将整个 optimizer step 捕获到一个 **CUDA Graph** 中，从而消除这一开销。

### 3.2 Hyperparameter Scaling

### 3.3 Stability Stress Test

## 4.Evaluation
