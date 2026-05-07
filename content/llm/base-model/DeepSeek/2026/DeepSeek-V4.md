# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

论文链接：

代码链接：https://huggingface.co/collections/deepseek-ai/deepseek-v4

## 摘要

我们发布了 DeepSeek-V4 系列的预览版，其中包括两个强大的混合专家 (MoE) 语言模型：**DeepSeek-V4-Pro**（参数量 1.6T，激活参数 49B）和 **DeepSeek-V4-Flash**（参数量 284B，激活参数 13B），两者均支持百万级上下文长度。DeepSeek-V4 系列在架构和优化方面进行了多项关键升级：(1) **混合注意力架构**，结合了压缩稀疏注意力 (CSA) 和重压缩注意力 (HCA)，以提高长上下文的效率；(2) **ManifoldConstrained Hyper-Connections (mHC)** ，增强了传统的残差连接；(3) 以及 **Muon 优化器**，以实现更快的收敛速度和更高的训练稳定性。我们使用超过 32T 的多样化高质量 token 对这两个模型进行了预训练，随后进行了全面的后训练流程，以释放并进一步增强它们的性能。 **DeepSeek-V4-Pro-Max** 是 DeepSeek-V4-Pro 的最高推理模式，重新定义了开放模型的性能标杆，在核心任务上超越了其前代产品。同时，DeepSeek-V4 系列在长上下文场景下也展现出极高的效率。在百万 Token 的上下文设置下，DeepSeek-V4-Pro 仅需 DeepSeek-V3.2 的 27% 单 Token 推理 FLOP 和 10% 的 KV 缓存。这使我们能够常规地支持百万 Token 的上下文，从而使长周期任务和进一步的测试时扩展变得更加可行。模型检查点可在 https://huggingface.co/collections/deepseek-ai/deepseek-v4 获取。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/0b99ef78d302415ab9575fdcc62a0831.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

推理模型的出现建立了一种新的测试时扩展范式，显著提升了大语言模型（LLM）的性能。然而，这种扩展范式从根本上受到原始注意力机制二次方计算复杂度的限制，这为超长上下文和推理过程造成了难以逾越的瓶颈。与此同时，从复杂的 Agent 工作流到大规模跨文档分析等长周期场景和任务的出现，也使得高效支持超长上下文成为未来发展的关键。尽管近期的开源项目提升了通用能力，但处理超长序列时的核心架构效率低下仍然是一个关键障碍，限制了测试时扩展的进一步提升，并阻碍了对长周期运行场景和任务的深入探索。

为了突破超长上下文的效率瓶颈，我们开发了 DeepSeek-V4 系列，其中包括参数量为 1.6T（激活参数 49B）的 **DeepSeek-V4-Pro** 预览版和参数量为 284B（激活参数 13B）的 **DeepSeek-V4-Flash** 预览版。通过架构创新，DeepSeek-V4 系列在处理超长序列的计算效率方面实现了质的飞跃。这一突破使得能够高效支持百万级 token 的上下文长度，开启了下一代 LLM 处理百万级上下文的新时代。我们相信，高效处理超长序列的能力将开启测试时间扩展的下一个前沿领域，为深入研究长周期任务铺平道路，并为探索在线学习等未来范式奠定必要的基础。

与 DeepSeek-V3 架构相比，DeepSeek-V4 系列保留了 DeepSeek-MoE 框架和多 token 预测 (MTP) 策略，并在架构和优化方面引入了多项关键创新。为了提高长上下文效率，我们设计了一种混合注意力机制，结合了 **Compressed Sparse Attention (CSA)** 和 **Heavily Compressed Attention (HCA)**。CSA 沿序列维度压缩 key-value 缓存，然后执行 DeepSeek Sparse Attention (DSA)，而 HCA 对 key-value 缓存应用更激进的压缩，但保持了密集注意力。为了增强建模能力，我们引入了 **Manifold-Constrained Hyper-Connections (mHC)**，以升级传统的残差连接。此外，我们将 **Muon 优化器**引入 DeepSeek-V4 系列的训练中，从而加快了收敛速度并提高了训练稳定性。

为了实现 DeepSeek-V4 系列的高效训练和推理以及高效的开发，我们引入了多项基础设施优化。首先，我们为 MoE 模块设计并实现了一个完全融合的**内核**，该内核涵盖了计算、通信和内存访问。其次，我们采用领域特定语言 (DSL) **TileLang** 来平衡开发效率和运行时效率。第三，我们提供高效的批处理不变且确定性的内核库，以确保训练和推理过程中位级可复现性。第四，我们为 MoE 专家权重和索引器 QK 路径引入了 **FP4 量化感知训练**，以减少内存和计算量。第五，在训练框架方面，我们扩展了自动微分框架，加入了**张量级检查点机制**，以实现细粒度的重计算控制；并通过混合 **ZeRO 策略**（用于 Muon 优化器）、基于重计算和融合内核的低成本 **mHC** 实现以及用于管理压缩注意力的**两阶段上下文并行性**来提升训练效率。最后，对于推理框架，我们设计了一种**异构 KV 缓存结构**，并采用磁盘存储策略来实现高效的共享前缀重用。

通过采用混合 CSA 和 HCA，并结合计算和存储方面的精度优化，DeepSeek-V4 系列相比 DeepSeek-V3.2 实现了显著更低的推理 FLOPs 和大幅缩减的 KV 缓存大小，尤其是在长上下文场景下。图 1 右侧展示了 DeepSeek-V3.2 和 DeepSeek-V4 系列的估计单 token 推理 FLOPs 和累积 KV 缓存大小。在 100 万 token 上下文场景下，即使是激活参数数量更多的 DeepSeek-V4-Pro，其单 token FLOPs（以等效 FP8 FLOPs 衡量）也仅为 DeepSeek-V3.2 的 27%，KV 缓存大小也仅为 DeepSeek-V3.2 的 10%。此外，DeepSeek-V4-Flash 激活的参数数量更少，效率进一步提升：在 100 万个 token 的上下文设置下，其单 token FLOPs 和 KV 缓存大小分别仅为 DeepSeek-V3.2 的 10% 和 7%。另外，**DeepSeek-V4 系列的路由专家参数采用 FP4 精度**。虽然在现有硬件上，FP4 × FP8 运算的峰值 FLOPs 与 FP8 × FP8 运算相同，但理论上，在未来的硬件上，其效率可以提升 1/3，这将进一步提高 DeepSeek-V4 系列的效率。

在预训练阶段，我们分别使用 32T 个 token 和 33T 个 token 对 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 进行训练。预训练完成后，这两个模型能够原生且高效地支持 1M 长度的上下文。在我们的内部评估中，DeepSeek-V4-Flash-Base 凭借其更高效的参数设计，在大多数基准测试中已经超越了 DeepSeek-V3.2-Base。DeepSeek-V4-Pro-Base 进一步扩展了这一优势，为 DeepSeek 基础模型树立了新的性能标杆，在推理、编码、长上下文和世界知识任务中均展现出全面优势。

DeepSeek-V4 系列的后训练流程采用两阶段范式：首先独立训练领域专家，然后通过 on-policy 蒸馏进行统一模型整合。首先，针对每个目标领域（例如数学、编程、智能体和指令执行），分别独立训练一个专家模型。基础模型首先在高质量的领域特定数据上进行有监督微调 (SFT)，以建立基础能力。随后，应用强化学习 (RL)，并采用组相对策略优化 (GRPO) 方法，进一步优化模型，使其行为与领域相符，并由针对特定成功标准定制的奖赏模型指导。此阶段产生一组各领域均表现卓越的专家。最后，为了整合这些不同的技能，通过 on-policy 蒸馏训练一个统一模型，该统一模型作为student 模型，与 teacher 模型一起学习，以优化反向 KL 损失。

**核心评估结果概要**
- **Knowledge**：在对广泛世界知识的评估中，DeepSeek-V4-Pro-Max（DeepSeek-V4-Pro 的最大推理模式）在 SimpleQA 和 Chinese-SimpleQA 基准测试中显著优于领先的开源模型。在教育知识方面（通过 MMLU-Pro、HLE 和 GPQA 评估），DeepSeek-V4-Pro-Max 略微领先于其开源同类模型。尽管在这些基于知识的评估中仍落后于领先的专有模型 Gemini-3.1-Pro，但 DeepSeek-V4-Pro-Max 与其之间的差距已显著缩小。
- **Reasoning**：通过扩展推理标记，DeepSeek-V4-Pro-Max 在标准推理基准测试中展现出优于 GPT-5.2 和 Gemini-3.0-Pro 的性能。然而，其性能略逊于 GPT-5.4 和 Gemini-3.1-Pro，表明其发展轨迹比目前最先进的模型落后约 3 至 6 个月。此外，DeepSeek-V4-Flash-Max 的性能与 GPT-5.2 和 Gemini-3.0-Pro 相当，证明其是一种适用于复杂推理任务的高性价比架构。
- **Agent**：在公开基准测试中，DeepSeek-V4-Pro-Max 与 Kimi-K2.6 和 GLM-5.1 等领先的开源模型性能相当，但略逊于前沿的闭源模型。在我们内部的评估中，DeepSeek-V4-Pro-Max 的性能优于 Claude Sonnet 4.5，并接近 Opus 4.5 的水平。
- **Long-Context**：DeepSeek-V4-Pro-Max 在 100 万个 token 的上下文窗口中，在合成和真实用例中均取得了强劲的成果，甚至在学术基准测试中超越了 Gemini-3.1-Pro。
- **DeepSeek-V4-Pro v.s. DeepSeek-V4-Flash**：由于参数规模较小，DeepSeek-V4-Flash-Max 在知识评估方面表现较差。然而，当分配更大的思维预算时，它在推理任务上取得了相当的成绩。在 Agent 评估中，尽管 DeepSeek-V4-Flash-Max 在多个基准测试中与 DeepSeek-V4-Pro-Max 的性能相当，但在更复杂、难度更高的任务上，它仍然落后于其规模更大的版本。

## 2.Architecture

<img
  src="https://i-blog.csdnimg.cn/direct/f1e82e0435304eedb60b12df9443ae52.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

总体而言，DeepSeek-V4 系列保留了 Transformer 架构和多 token 预测 (MTP) 模块，并在 DeepSeek-V3 的基础上进行了多项关键升级：(1) 首先，我们引入了 **Manifold-Constrained Hyper-Connections (mHC)** 来增强传统的残差连接；(2) 其次，我们设计了一种**混合注意力架构**，通过压缩稀疏注意力和重压缩注意力显著提升了长上下文效率；(3) 第三，我们采用 **Muon** 作为优化器。对于混合专家 (MoE) 组件，我们仍然沿用 DeepSeek-MoE 架构，仅对 DeepSeek-V3 进行了细微调整。多 token 预测 (MTP) 配置与 DeepSeek-V3 完全相同。所有其他未明确说明的细节均遵循 DeepSeek-V3 中的设置。图 2 展示了 DeepSeek-V4 的整体架构，具体细节如下所述。

### 2.1 Designs Inherited from DeepSeek-V3

**Mixture-of-Experts**。与之前的 DeepSeek 系列模型一样，DeepSeek-V4 系列也采用了 DeepSeekMoE 范式构建前馈网络 (FFN)，该范式设置了细粒度的路由专家和共享专家。与 DeepSeek-V3 不同的是，我们将计算亲和力得分的激活函数从 $Sigmoid(·)$ 改为 $Sqrt(Softplus(·))$。为了实现负载均衡，我们还采用了无辅助损失策略，并辅以轻微的序列级均衡损失，以防止单个序列内部出现极端的不平衡。对于 DeepSeek-V4，我们**取消了对路由目标节点数量的限制**，并精心重新设计了并行策略以保持训练效率。此外，与 DeepSeek-V3 相比，我们将初始几个 Transformer 模块中的密集 FFN 层替换为采用**哈希路由的 MoE 层**。哈希路由策略根据预定义的哈希函数，结合输入 token ID 来确定每个 token 的目标专家。

**Multi-Token Prediction**。与 DeepSeek-V3 一样，DeepSeek-V4 系列也设置了 MTP 模块和目标。鉴于 MTP 策略已在 DeepSeek-V3 中得到验证，我们在 DeepSeek-V4 系列中沿用了相同的策略，未作任何修改。

### 2.2 Manifold-Constrained Hyper-Connections

如图 2 所示，DeepSeek-V4 系列引入了 Manifold-Constrained Hyper-Connections (mHC) 来增强相邻 Transformer 模块之间的传统残差连接。**与朴素超连接 (HC) 相比，mHC 的核心思想是将残差映射约束到特定的流形上，从而在保持模型表达能力的同时，增强信号在各层间的传播稳定性**。本小节将简要介绍标准超连接，并阐述我们如何设计 mHC 以实现稳定的训练。

**Standard Hyper-Connections**。标准 HC 将残差流的宽度扩展了 $n_{hc}$ 倍。具体来说，残差流的形状从 $\mathbb R^d$ 扩展到 $\mathbb R^{n_{hc}×d}$，其中 $d$ 是实际层输入的隐藏层大小。令 $X_l = [x_{l,1}; ...; x_{l,n_{hc}}]^T ∈ \mathbb R^{n_{hc}×d}$ 为第 $l$ 层之前的残差状态。**HC 引入了三个线性映射**：输入映射 $A_l ∈ \mathbb R^{1×n_{hc}}$，残差变换 $B_l ∈ \mathbb R^{n_{hc}×n_{hc}}$，以及输出映射 $C_l ∈ \mathbb R^{n_{hc}×1}$。残差状态的更新公式如下：

```math
X_{l+1}=B_lX_l+C_l\mathcal F_l(A_lX_l),\tag{1}
```

其中 $\mathcal F_l$ 表示第 $l$ 层（例如，MoE 层），其输入和输出形状均为 $\mathbb R^d$。需要注意的是，实际的层输入 $A_lX_l ∈ \mathbb R^d$ 也是 $d$ 维的，因此扩展残差宽度不会影响内部层的设计。**HC 将残差宽度与实际隐藏层大小解耦，提供了一个计算开销极小的互补缩放轴，因为 $n_{hc}$ 通常远小于隐藏层大小 $𝑑$**。然而，尽管 HC 已展现出提升模型性能的潜力，但我们发现，当堆叠多层时，训练过程经常会出现数值不稳定性，这阻碍了 HC 的扩展。

**Manifold-Constrained Residual Mapping**。mHC 的核心创新在于将残差映射矩阵 $B_l$ 限制在双随机矩阵流形（伯克霍夫多面体）$\mathcal M$ 上，从而增强信号跨层传播的稳定性：

```math
B_l\in\mathcal M:=\{M\in\mathbb R^{n\times n}|M\textbf 1_n=\textbf 1_n,\textbf 1^T_nM=\textbf 1^T_n,M\ge 0\}\tag{2}
```

该约束确保映射矩阵 $||B_l||_2$ 的谱范数小于等于 1，从而使残差变换非扩张，提高了前向传播和反向传播过程中的数值稳定性。此外，集合 $M$ 在乘法运算下封闭，保证了在 mHC 深度堆叠场景下的稳定性。另外，输入变换 $A_l$ 和输出变换 $C_l$ 也通过Sigmoid函数约束为非负且有界，以**避免信号抵消**的风险。

**Dynamic Parameterization**。三个线性映射的参数是动态生成的，它们被分解为动态（输入相关）分量和静态（输入无关）分量。给定输入 $X_l ∈ \mathbb R^{n_{hc}×d}$，首先对其进行展平和归一化：$\hat X_l = RMSNorm(vec(X_l)) ∈ \mathbb R^{1×n_{hc}d}$。然后，我们遵循传统的 HC 方法生成无约束的原始参数 $\tilde A_l ∈ \mathbb R^{1×n_{hc}}$，$\tilde B_l ∈ \mathbb R^{n_{hc}×n_{hc}}$，以及 $\tilde C_l ∈ \mathbb R^{n_{hc}×1}$：

```math
\tilde A_l=\alpha^{pre}_l\cdot(\hat X_lW^{pre}_l)+S^{pre}_l,\tag{3}
```

```math
\tilde B_l=\alpha^{res}_l\cdot Mat(\hat X_lW^{res}_l)+S^{res}_l,\tag{4}
```

```math
\tilde C_l=\alpha^{post}_l\cdot(\hat X_lW^{post}_l)^T+S^{post}_l,\tag{5}
```

其中，$W^{pre}_l, W^{post}_l ∈ \mathbb R^{n_{hc}d×n_{hc}}$ 和 $W^{res}_l ∈ \mathbb R^{n_{hc}d×n^2_{hc}}$ 是用于生成动态分量的可学习参数；$Mat(·)$ 将大小为 $1 × n^2_{hc}$ 的向量重塑为大小为 $n_{hc} × n_{hc}$ 的矩阵；$S^{pre}_l ∈ \mathbb R^{1×𝑛_{hc}}$，$S^{post}_l ∈ \mathbb R^{n_{hc}×1}$，$S^{res}_l ∈ \mathbb R^{n_{hc}×n_{hc}}$ 是可学习的静态偏置；其中 $\alpha^{pre}_l,\alpha^{res}_l,\alpha^{post}_l∈ \mathbb R$ 是可学习的门控因子，初始值为较小的值。

**Applying Parameter Constraints**。在获得无约束的原始参数 $\tilde{A}_l, \tilde{B}_l, \tilde{C}_l$ 之后，我们会对它们施加前文所述的约束，以增强数值稳定性。具体来说，对于输入映射和输出映射，我们使用 Sigmoid 函数 $\sigma(\cdot)$ 来确保它们的非负性和有界性：

```math
A_l = \sigma(\tilde{A}_l),\tag{6}
```

```math
C_l = 2\sigma(\tilde{C}_l).\tag{7}
```

至于残差映射 $\tilde{B}_l$，我们将其投影到双随机矩阵流形 $\mathcal{M}$ 上。这通过 Sinkhorn-Knopp 算法实现：该算法首先对 $\tilde{B}_l$ 应用指数函数以确保正性，得到 $M^{(0)} = \exp(\tilde{B}_l)$，然后迭代地进行列归一化和行归一化：

```math
M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)})),\tag{8}
```

其中 $\mathcal{T}_r$ 和 $\mathcal{T}_c$ 分别表示行归一化和列归一化。该迭代会收敛到一个受约束的双随机矩阵 $B_l = M^{(t_{\max})}$。我们选择 $t_{\max}=20$ 作为一个实用取值。

### 2.3 Hybrid Attention with CSA and HCA

当上下文长度达到极高规模时，注意力机制成为模型中的主要计算瓶颈。针对 DeepSeek-V4，我们设计了两种高效的注意力架构—— **Compressed Sparse Attention (CSA)** 和 **Heavily Compressed Attention (HCA)** ——并采用它们的交错混合配置，从而显著降低了长文本场景下注意力机制的计算成本。CSA 融合了压缩和稀疏注意力策略：首先将每 $m$ 个 token 的 key-value 缓存压缩成一个条目，然后应用 **DeepSeek Sparse Attention (DSA)**，其中每个 query token 仅关注 k 个压缩后的 KV 条目。HCA 则通过将每 $m' (≫ m)$ 个 token 的 KV 缓存合并成一个条目来实现极致压缩。CSA 和 HCA 的混合架构显著提升了 DeepSeek-V4 系列的长上下文处理效率，使得处理百万级 token 的上下文在实践中成为可能。本小节描述了我们混合注意力架构的核心技术，我们还提供了一个开源实现 (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)，以便更明确地说明更多细节。

#### 2.3.1 Compressed Sparse Attention

<img
  src="https://i-blog.csdnimg.cn/direct/72f91d5e66994a9ea1b13e3e5ecffd45.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

CSA 的核心架构如图 3 所示，它首先将每 $m$ 个 token 的 KV 缓存压缩成一个条目，然后应用 DeepSeek Sparse Attention 进行进一步加速。

**Compressed Key-Value Entries**。设 $H ∈ \mathbb R^{n×d}$ 为输入隐藏状态序列，其中 $n$ 为序列长度，$d$ 为隐藏层大小。CSA 首先计算两组 KV 条目 $C^a, C^b ∈ \mathbb R^{n×c}$ 及其对应的压缩权重 $Z_a, Z_b ∈ \mathbb R^{n×c}$，其中 $c$ 为头的维度：

```math
C^a=H\cdot W^{aKV},\quad C^b=H\cdot W^{bKV},\tag{9}
```

```math
Z^a=H\cdot W^{aZ},\quad Z^b=H\cdot W^{bZ}\tag{10}
```

其中，$W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ} ∈ \mathbb R^{d×c}$ 为可训练参数。接下来，$C^a$ 和 $C^b$ 中的每个 $m$ 个 KV 条目将根据其压缩权重和可学习的位置偏差 $B^a, B^b ∈ \mathbb R^{𝑚×𝑐}$ 压缩成一个条目，生成 $C^{Comp} ∈ \mathbb R^{\frac{n}{m}×c}$。每个压缩条目 $C^{Comp}_i∈ \mathbb R^c$ 由以下公式计算：

```math
[S^a_{mi:m(i+1)-1};S^b_{m(i-1):mi-1}]=Softmax_{row}([Z^a_{mi:m(i+1)-1}+B^a;z^b_{m(i-1):mi-1}+B^b]),\tag{11}
```

```math
C^{Comp}_i=\sum^{m(i+1)-1}_{j=mi}S^a_j\odot C^a_j+\sum^{mi-1}_{j=m(i-1)}S^b_j\odot C^b_j\tag{12}
```

其中 $⊙$ 表示 Hadamard 积；Softmaxrow(·) 表示沿行维度的 softmax 操作，它对来自 $Z^a$ 和 $Z^b$ 的 2𝑚 个元素进行归一化。当 $i = 0$ 时，$Z^b_{m(i−1):mi−1}$ 用负无穷大填充，$C^b_{m(i−1):mi−1}$ 用零填充。注意，每个 $C^{Comp}_i$ 都源自 $2m$ 个 KV 条目，但用于 $C^{Comp}_i$ 的 $C^b$ 索引和用于 $C^{Comp}_{i−1}$ 的 $C^a$ 索引是重叠的。因此，CSA 实际上将序列长度压缩了 $\frac{1}{m}$ 倍。

**Lightning Indexer for Sparse Selection**。在获得压缩后的 KV 项 $C^{Comp}$ 后，CSA 应用 DSA 策略选择 top-k 个压缩后的 KV 项作为核心注意力机制。首先，CSA 执行与 $C^{Comp}$ 相同的压缩操作，得到压缩后的索引器 key $K^{IComp} ∈ \mathbb R^{\frac{n}{m}×c^I}$，其中 $c^I$ 是索引器头的维度。然后，对于 query token $t$，我们以低秩方式生成索引器 query $\{\textbf q^𝐼_{t,1}; \textbf q^I_{𝑡,2};...; \textbf q^I_{i,n^I_h}\}$：

```math
\textbf c^Q_t=\textbf h_t\cdot W^{DQ},\tag{13}
```

```math
[\textbf q^I_{t,1};\textbf q^I_{t,2};...;\textbf q^I_{t,n^I_h}]=\textbf q^I_t=c^Q_t\cdot W^{IUQ},\tag{14}
```

其中 $\textbf h_t ∈ \mathbb R^d$ 是 query token $𝑡$ 的输入隐藏状态；$c^Q_t ∈ \mathbb R^{d_c}$ 是 query 的压缩潜在向量；$d_c$ 表示 query 压缩维度；$n^I_h$ 表示索引器 query 头的数量；$W^{DQ} ∈ \mathbb R^{d×d_c}$ 和 $W^{IUQ} ∈ \mathbb R^{d_c×c^In^I_h}$ 分别是索引器 query 的下投影矩阵和上投影矩阵。接下来，计算 query token $t$ 与其前一个压缩块 $s$ ($s < Floor(\frac{t}{m})$) 之间的索引得分 $I_{t,s} ∈ \mathbb R$，方法是：

```math
[w^I_{t,1};w^I_{t,2};...;w^I_{t,n^I_h}]=\textbf w^I_t=\textbf h_t\cdot W^w,\tag{15}
```

```math
I_{t,s}=\sum^{n^I_h}_{h=1}w^I_{t,h}\cdot ReLU(q^I_{t,h}\cdot K^{IComp}_s),\tag{16}
```

其中 $W^w ∈ \mathbb R^{d×n^I_h}$ 是一个可学习矩阵；$w^I_{t,h}∈ \mathbb R$ 是第 $h$ 个索引器头的权重。对于query 词元 $t$，给定其索引得分 $I_{t,:}$，我们采用 top-k 个选择器来选择性地保留压缩 KV 条目子集 $C^{SprsComp}_t$，以便进行后续的核心注意力机制：

```math
C^{SprsComp}_t=\{C^{Comp}_s|I_{t,s}\in Top-k(I_{t,:})\}.\tag{17}
```

**Shared Key-Value MQA**。在选择稀疏 KV 条目之后，CSA 随后以 Multi-Query Attention（MQA）的方式执行核心注意力，其中 $C_t^{\mathrm{SprsComp}}$ 中的每个压缩 KV 条目同时作为注意力的 key 和 value。具体来说，对于 query token $t$，我们首先从压缩潜在向量 $c_t^Q$ 中生成注意力查询 $\{\textbf q_{t,1}; \textbf q_{t,2}; ...; \textbf q_{t,n_h}\}$：

```math
[\textbf q_{t,1}; \textbf q_{t,2}; ...; \textbf q_{t,n_h}] = \textbf q_t = \textbf c_t^Q \cdot W^{UQ},\tag{18}
```

其中，$n_h$ 表示 query 头的数量；$W^{UQ} \in \mathbb{R}^{d_c \times c n_h}$ 是用于 query 的上投影矩阵。注意，潜在 query 向量 $c_t^Q$ 与用于索引器 query 的向量是共享的。接下来，我们在 ${q_{t,i}}$ 和 $C_t^{\mathrm{SprsComp}}$ 上执行 MQA：

```math
o_{t,i} = \mathrm{CoreAttn}\left(\mathrm{query}=q_{t,i},\mathrm{key}=C_t^{\mathrm{SprsComp}},\mathrm{value}=C_t^{\mathrm{SprsComp}}\right),\tag{19}
```

其中，$o_{t,i} \in \mathbb{R}^{c}$ 是第 $i$ 个头在第 $t$ 个 token 处的核心注意力输出；$\mathrm{CoreAttn}(\cdot)$ 表示核心注意力操作。

**Grouped Output Projection**。在 DeepSeek-V4 的配置中，$cn_h$ 相当大。因此，直接将核心注意力操作的输出 $[\textbf o_{t,1}; \textbf o_{t,2}; ...; \textbf o_{t,n_h}] = \textbf o_t \in \mathbb{R}^{cn_h}$ 投影到一个 $d$ 维隐藏状态，会带来显著的计算负担。为了降低这一成本，我们设计了一种分组输出投影策略。具体来说，我们首先将 $n_h$ 个输出划分为 $g$ 个组，然后对于每一组输出 $o_{t,i}^{G} \in \mathbb{R}^{c\frac{n_h}{g}}$，将其投影到一个 $d_g$ 维的中间输出 $o_{t,i}^{G'} \in \mathbb{R}^{d_g}$，其中 $d_g < c\frac{n_h}{g}$。最后，我们将中间输出 $[o_{t,1}^{G'}; o_{t,2}^{G'}; ...; o_{t,g}^{G'}] \in \mathbb{R}^{d_g g}$ 投影到最终的注意力输出 $\hat{o}_t \in \mathbb{R}^{d}$。

#### 2.3.2 Heavily Compressed Attention

<img
  src="https://i-blog.csdnimg.cn/direct/a488aa72b37e4569bd16de3e4d4575d5.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

HCA 的核心架构如图 4 所示，它以更重的方式压缩 KV 缓存，但不采用稀疏注意力机制。

**Compressed Key-Value Entries**。总体而言，HCA 的压缩策略与 CSA 类似，但采用更大的压缩率 $m' (≫ m)$，并且**不进行重叠压缩**。设 $H ∈ \mathbb R^{n×d}$ 为输入隐藏状态序列，HCA 首先计算原始 KV 条目 $C ∈ \mathbb R^{n×c}$ 及其对应的压缩权重 $Z ∈ \mathbb R^{n×c}$：

```math
C=H\cdot W^{KV},\tag{20}
```

```math
Z=H\cdot W^Z,\tag{21}
```

其中 $W^{KV}, W^Z ∈ \mathbb R^{d×c}$ 为可训练参数。接下来，根据压缩权重和可学习的位置偏差 $B ∈ \mathbb R^{m'×c}$，将 $C$ 中的每 $m'$ 个 KV 条目压缩为一个条目，生成 $C^{Comp} ∈ \mathbb R^{\frac{n}{m'} ×c}$。每个压缩条目 $C^{Comp}_i ∈ \mathbb R^c$ 的计算方式如下：

```math
S_{m'i:m'(i+1)-1}=Softmax_{row}(Z_{m'i:m'(i+1)-1}+B),\tag{22}
```

```math
C^{Comp}_i=\sum^{m'(i+1)-1}_{j=m'i}S_j\odot C_j.\tag{23}
```

通过这种压缩操作，HCA 将序列长度压缩到 $\frac{1}{m'}$ 倍。

**Shared Key-Value MQA and Grouped Output Projection**。HCA 也像 CSA 一样采用了共享 KV MQA 和分组输出投影策略。在 KV 压缩之后，对于 query token $t$，HCA 首先以低秩方式生成注意力 query $\{\textbf q_{t,1}; \textbf q_{t,2}; ...; \textbf q_{t,n_h}\}$：

```math
c^Q_t=\textbf h_t\cdot W^{DQ},\tag{24}
```

```math
[\textbf q_{t,1};\textbf q_{t,2};...;\textbf q_{t,n_h}]=\textbf q_t=\textbf c^Q_t\cdot W^{UQ},\tag{25}
```

其中 $\textbf h_t ∈ \mathbb R^d$ 是 query token $t$ 的输入隐藏状态；$n_h$ 表示 query 头的数量；$W^{DQ} ∈ \mathbb R^{d×d_c}$ 和 $W^{UQ} ∈ \mathbb R^{d_c×cn_h}$ 分别是 query 的下投影矩阵和上投影矩阵。接下来，我们对 $\{\textbf q_{t,i}\}$ 和 $C^{Comp}$ 执行 MQA：

```math
\textbf o_{i,t}=CoreAttn(query=\textbf q_{t,i},key=C^{Comp},value=C^{Comp}),\tag{26}
```

其中 $\textbf o_{t,i} ∈ \mathbb R^c$ 是第 $i$ 个头在第 $t$ 个 token 处的核心注意力输出。接下来，与 CSA 类似，HCA 将 $n_h$ 个输出分成 $g$ 个组，对于每个输出组 $\textbf o^{G'}_{t,i} ∈ \mathbb R^{c\frac{n_h}{g}}$，HCA 将其投影到一个 $d_g$ 维的中间输出 $\textbf o^{G'}_{t,i} ∈ \mathbb R^{d_g}$，其中 $d_h < c\frac{n_h}{g}$。最后，HCA 投影中间输出 $[\textbf o^{G'}_{t,1}; \textbf o^{G'}_{t,2}; ...; \textbf o^{G'}_{t,g}] ∈ \mathbb R^{d_gg}$ 到最终注意力输出 $\hat {\textbf o}_t ∈ \mathbb R^d$。

#### 2.3.3 Other Details

除了上文所述的 CSA 和 HCA 核心架构之外，我们的混合注意力机制还融合了其他几种技术。为了行文清晰，我们在上文引言中省略了这些技术，并在本小节中对其进行简要介绍。此外，**本小节仅关注这些技术的核心思想，为了简洁起见，可能会省略一些细节**。我们建议读者参考我们的开源实现以获取更详细的信息。

**Query and Key-Value Entry Normalization**。对于 CSA 和 HCA，我们在核心注意力操作之前，对 query 的每个头以及压缩 key-value 对的唯一头执行额外的 RMSNorm 操作。这种归一化操作可以避免注意力 logtis 爆炸，并可能提高训练稳定性。

**Partial Rotary Positional Embedding**。对于 CSA 和 HCA，我们将旋转位置嵌入（RoPE）部分应用于注意力 query、KV 项以及核心注意力输出。**具体来说，对于 CSA 和 HCA 中使用的每个 query 向量和 KV 向量，我们将 RoPE 应用于其最后 64 个维度**。由于 KV 项同时作为注意力的 key 和 value，原生的核心注意力输出 $\{\textbf o_{t,i}\}$ 会携带绝对位置嵌入，这是由 KV 项的加权和得到的。作为一种应对措施，我们还在每个 $\{\textbf o_{t,i}\}$ 的最后 64 个维度上应用位置为 $-i$ 的 RoPE。通过这种方式，核心注意力的输出也会携带相对位置嵌入——每个 KV 项对核心注意力输出的贡献也将与 query 和 KV 条目之间的距离相关。

**Additional Branch of Sliding Window Attention**。为了在 CSA 和 HCA 中严格保持因果性，每个 query 只关注其之前的压缩 KV 块。因此，一个 query 无法访问其自身压缩块内其他 token 的信息。同时，在语言建模中，较近的 token 通常与 query token 具有更高的相关性。基于这些原因，我们在 CSA 和 HCA 中引入了一个补充性的注意力分支，以滑动窗口的方式更好地建模局部依赖关系。具体来说，对于每个 query token，我们额外生成 $n_{\text{win}}$ 个未压缩的 KV 条目，对应最近的 $n_{\text{win}}$ 个 token。在 CSA 和 HCA 的核心注意力中，滑动窗口中的这些 KV 项将与压缩 KV 条目一起使用。

**Attention Sink**。在 CSA 和 HCA 的核心注意力中，我们采用了 attention sink 技巧。具体来说，我们设置了一系列**可学习的** sink logits $\{z'_1, z'_2, ..., z'_{n_h}\}$。对于第 $h$ 个注意力头，$\mathrm{Exp}(z'_h)$ 会被加到注意力分数分母中：

```math
s_{h,i,j} =
\frac{\mathrm{Exp}(z_{h,i,j})}
{\sum_k \mathrm{Exp}(z_{h,i,k}) + \mathrm{Exp}(z'_h)},
\tag{27}
```

其中 $s_{h,i,j}, z_{h,i,j} \in \mathbb{R}$ 分别表示第 $h$ 个注意力头中，第 $i$ 个查询 token 与第 $j$ 个前置 token 或压缩块之间的注意力分数和注意力 logit。该技术允许每个 Query 头将其总注意力分数调整为不等于 1，甚至可以接近 0。

#### 2.3.4 Efficiency Discussion

由于采用了混合 CSA 和 HCA，并结合低精度计算与存储，DeepSeek-V4 系列的注意力模块在注意力 FLOPs 和 KV cache 大小两个方面都实现了显著的效率提升，尤其是在长上下文场景中。首先，**我们对 KV 项采用混合存储格式：BF16 精度用于旋转位置嵌入（RoPE）维度，而 FP8 精度用于其余维度**。与纯 BF16 存储相比，这种混合表示几乎将 KV cache 大小减少了一半。其次，**lightning indexer 内部的注意力计算以 FP4 精度执行**，这加速了极长上下文下的注意力操作。第三，相较于 DeepSeek-V3.2，DeepSeek-V4 系列选择了更小的 attention top-k，从而提升了模型在短文本和中等长度文本上的效率。最后，也是最重要的是，压缩注意力和混合注意力技术显著降低了 KV cache 大小和计算 FLOPs。

以 BF16 GQA8 且 head dimension 为 128 作为基线——这是 LLM 注意力的一种常见配置——在 1M 上下文设置下，DeepSeek-V4 系列的 KV cache 大小可以被大幅降低到该基线的大约 2%。此外，即使与 DeepSeek-V3.2 相比——后者已经是一个高效的基线——DeepSeek-V4 系列仍然展现出显著的效率优势。二者在推理 FLOPs 和 KV cache 大小方面的比较见图 1 右侧。

### 2.4 Muon Optimizer

<img
  src="https://i-blog.csdnimg.cn/direct/5cbaed3f011f4f03a2b02fefe9a45de0.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

由于 Muon 优化器收敛速度更快、训练稳定性更高，因此我们在 DeepSeek-V4 系列的大部分模块中都采用了该优化器。Muon 优化器的完整算法概述于算法 1 中。

**Basic Configurations**。我们对 embedding 模块、预测头模块、mHC 模块的静态偏差和门控因子以及所有 RMSNorm 模块的权重均采用 AdamW 优化器。所有其他模块均使用 Muon 进行更新。参照 Liu et al. (2025) 的方法，我们也对 Muon 参数应用了权重衰减，使用了 Nesterov 技巧，并对更新矩阵的均方根 (RMS) 进行了重新缩放，以便重复利用我们的 AdamW 超参数。与他们不同的是，我们使用混合牛顿-舒尔茨迭代法进行正交化。

**Hybrid Newton-Schulz Iterations**。 对于给定矩阵 $M$，设其奇异值分解（SVD）为 $M = U\Sigma V^T$。Newton-Schulz 迭代的目标是将 $M$ 近似正交化为 $UV^T$。通常，会先将 $M$ 归一化为 $M_0 = M / |M|_F$，以确保其最大奇异值不超过 1。随后，每次 Newton-Schulz 迭代执行如下操作：

```math
M_k = aM_{k-1} + b(M_{k-1}M_{k-1}^T)M_{k-1} + c(M_{k-1}M_{k-1}^T)^2M_{k-1}.
\tag{28}
```

我们的混合 Newton-Schulz 方法在两个不同阶段中执行 10 次迭代。在前 8 步中，我们使用系数 $(a,b,c) = (3.4445, -4.7750, 2.0315)$ 来推动快速收敛，使奇异值接近 1。在最后 2 步中，我们切换到系数 $(a,b,c) = (2, -1.5, 0.5)$，从而将奇异值精确稳定在 1。

**Avoiding Exploding Attention Logits**。DeepSeek-V4 系列的注意力架构允许我们直接在注意力 query 和 KV 项上应用 RMSNorm，这有效防止了注意力 logits 爆炸。因此，我们在 Muon 优化器中没有采用 QK-Clip 技术。

## 3.General Infrastructures

### 3.1 Fine-Grained Communication-Computation Overlap in Expert Parallelism

<img
  src="https://i-blog.csdnimg.cn/direct/2cd5758cae2d4bfeb8ad274e7e1e43cd.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

专家混合模型（MoE）可以通过专家并行（EP）加速。然而，EP需要复杂的节点间通信，并且对互连带宽和延迟提出了很高的要求。为了缓解 EP 中的通信瓶颈，并在较低的互连带宽要求下实现更高的端到端性能，我们提出了一种细粒度的 EP 方案，**该方案将通信和计算融合到一个流水线内核中，以实现通信与计算的重叠**。

**Communication Latency Can Be Hidden**。我们 EP 方案的关键在于，通信延迟可以有效地隐藏在 MoE 层的计算之下。如图 5 所示，在DeepSeek-V4系列中，**每个 MoE 层主要可分解为四个阶段**：两个通信密集型阶段（Dispatch 和 Combine）以及两个计算密集型阶段（Linear-1 和 Linear-2）。我们的性能分析表明，在单个 MoE 层内，通信总时间小于计算时间。因此，在将通信和计算融合到一个统一的流水线后，计算仍然是主要的瓶颈，这意味着系统可以在不降低端到端性能的情况下容忍较低的互连带宽。

**Fine-Grained EP Scheme**。为了进一步降低互连带宽需求并放大重叠的优势，我们引入了一种更细粒度的专家划分方案。受诸多相关工作的启发，我们将专家分成若干波次进行调度。每波次包含一小部分专家。一旦波次内的所有专家完成通信，即可立即开始计算，无需等待其他专家。**在稳定状态下，当前波次的计算、下一波次的 token 转移以及已完成专家的结果发送均并行进行**，如图 5 所示。这在专家之间形成了一个细粒度的流水线，确保了整个波次内计算和通信的连续性。基于波次的调度方案能够显著提升强化学习 (RL) 部署等极端情况下的性能，因为这类部署通常会遇到长尾小批量数据。

**Performance and Open-Sourced Mega-Kernel**。我们已在 NVIDIA GPU 和华为昇腾 NPU 平台上验证了细粒度 EP 方案。与强大的非融合基线方案相比，该方案在一般推理工作负载下实现了 1.50 至 1.73 倍的加速，在诸如强化学习部署和高速 Agent 服务等对延迟敏感的场景下，加速比最高可达 1.96 倍。我们已将基于 CUDA 的超内核实现 **MegaMoE** 作为 DeepGEMM 的一个组件开源。

**Observations and Proposals**。我们分享了一些来自 kernel 开发的观察和经验，并向硬件厂商提出若干建议，希望有助于高效硬件设计，并实现更好的软硬件协同设计：

* **计算-通信比。** 完整的通信-计算重叠取决于计算-通信比，而不仅仅取决于带宽本身。将峰值计算吞吐量记为 $C$，互连带宽记为 $B$，当 $C/B \leq V_{\text{comp}}/V_{\text{comm}}$ 时，通信可以被完全隐藏，其中 $V_{\text{comp}}$ 表示计算量，$V_{\text{comm}}$ 表示通信量。对于 DeepSeek-V4-Pro，每个 token-expert 对需要 $6hd$ FLOPs（SwiGLU 的 gate、up 和 down projections），但只需要 $3h$ 字节的通信量（FP8 Dispatch + BF16 Combine），因此可简化为：

```math
\frac{C}{B} \leq 2d = 6144\ \text{FLOPs/Byte}.
```

也就是说，每 1 GBps 的互连带宽足以隐藏 6.1 TFLOP/s 计算所需的通信。一旦带宽达到这一阈值，它就不再是瓶颈；继续投入额外硅面积来进一步提升带宽会带来递减收益。我们鼓励未来的硬件设计应瞄准这种平衡点，而不是无条件地扩展带宽。

* **功耗预算。** 极致的 kernel fusion 会同时将计算、内存和网络推到高负载状态，使功耗限制成为关键的性能瓶颈。我们建议未来的硬件设计为这类完全并发的工作负载提供足够的功耗余量。

* **通信原语。** 我们采用了一种基于 pull 的方法，即每个 GPU 主动从远端 GPU 读取数据，从而避免细粒度 push 所带来的高通知延迟。未来，如果硬件能够提供更低延迟的跨 GPU 信令，push 将变得可行，并支持更自然的通信模式。

* **激活函数。** 我们建议用一种低成本的逐元素激活函数替代 SwiGLU，该激活函数不涉及指数或除法运算。这会直接减轻 GEMM 后处理的负担；并且在相同参数预算下，移除 gate projection 会扩大中间维度 $d$，进一步放宽带宽需求。

### 3.2 Flexible and Efficient Kernel Development with TileLang

实际上，我们精心设计的模型架构会产生数百个细粒度的 Torch ATen 算子。**我们采用 TileLang 开发了一组融合内核来替换其中绝大多数算子，从而以最小的投入实现最佳性能**。它还允许我们在验证过程中快速构建诸如注意力机制变体之类的算子原型。这些内核在模型架构开发、大规模训练以及最终的推理服务生产部署中都发挥着至关重要的作用。作为一种领域特定语言 (DSL)，TileLang 兼顾了开发效率和运行时性能，既能实现快速开发，又能在同一代码库内支持深度迭代优化。此外，我们与 TileLang 社区紧密合作，以促进更敏捷、高效和稳定的内核开发工作流程。

**Reducing Invocation Overhead with Host Codegen**。随着加速器性能持续提升，CPU 侧的编排开销变得越来越突出。对于小型且高度优化的 kernel，这类固定的 host 开销很容易限制利用率和吞吐量。该开销的一个常见来源是：host 侧逻辑，例如运行时契约检查，通常为了灵活性而使用 Python 编写，因此会产生固定的单次调用成本。

我们通过 **Host Codegen** 缓解这一开销，它将大部分 host 侧逻辑移动到生成的 host 代码中。具体来说，我们首先在 IR（Intermediate Representation，中间表示）层面共同生成 device kernel 和轻量级 host launcher，并嵌入从语言前端解析出的必要元数据——例如数据类型、rank/shape 约束，以及 stride/layout 假设。随后，launcher 会被 lower 到基于 TVM-FFI 框架构建的 host 源代码中；该框架紧凑的调用约定和零拷贝 tensor 互操作共同最小化了 host 侧开销。在运行时，这些生成的 host 代码会执行验证和参数封送，从而将所有单次调用检查移出 Python 执行路径。我们的测量结果表明，CPU 侧验证开销从数十或数百微秒降低到每次调用不到 1 微秒。

**SMT-Solver-Assisted Formal Integer Analysis**。TileLang kernel 涉及复杂的 tensor 索引算术，因此需要强大的形式化整数分析。在布局推断、内存冲突检测和边界分析等编译阶段中，编译器必须验证整数表达式是否满足特定性质，从而启用相应优化。因此，更强的形式化分析能力可以解锁更高级、更复杂的优化机会。

为此，我们将 Z3 SMT 求解器集成到 TileLang 的代数系统中，为 tensor 程序中的大多数整数表达式提供形式化分析能力。我们通过将 TileLang 的整数表达式转换为 Z3 的无量词非线性整数算术（QF_NIA），在计算开销和形式表达能力之间取得平衡。基于整数线性规划（ILP）求解器，QF_NIA 可以无缝处理 kernel 中常见的标准线性整数表达式。此外，其内在的非线性推理能力能够有效应对更高级的挑战，例如可变 tensor shape 下的向量化。在合理的资源限制下，Z3 提升了整体优化性能，同时将编译时间开销限制在仅几秒以内。其影响在多个 pass 中都非常显著，包括向量化、barrier 插入和代码简化。

**Numerical Precision and Bitwise Reproducibility**。在生产环境中，数值的正确性和可复现性与原始吞吐量同等重要。因此，我们默认优先考虑精度：编译器层面禁用快速数学优化，并且仅以显式可选的前端运算符（例如，T.__exp、T.__log 和 T.__sin）的形式提供影响精度的近似值。相反，当需要严格的 IEEE-754 语义时，TileLang 提供符合 IEEE 标准的内部函数，并带有显式的舍入模式（例如，T.ieee_fsqrt、T.ieee_fdiv 和 T.ieee_add），使开发人员能够精确地指定数值行为。

我们还致力于实现位级可复现性，以便将内核与手写 CUDA 基线进行验证。我们使 TileLang 的代数简化和降阶规则与主流 CUDA 工具链（例如 NVCC）保持一致，以避免引入意外的位级差异。布局注释（例如 T.annotate_layout）进一步允许用户确定与布局相关的降阶决策，从而保持求值和累加顺序与参考 CUDA 实现一致，并在需要时实现位级完全相同的输出。

我们的评估表明，这些以准确性和可重复性为导向的设计选择不会牺牲性能：在保守的默认值下，TileLang 内核仍然具有竞争力，同时提供了调整选项，可以有选择地放宽数值约束以获得更高的速度。

### 3.3 High-Performance Batch-Invariant and Deterministic Kernel Libraries

为了实现高效的训练与推理，我们开发了一整套高性能计算 kernel。除了提供基础功能并最大化硬件利用率之外，另一个关键设计目标是确保训练的可复现性，以及在预训练、后训练和推理流水线之间实现 bitwise alignment。因此，我们实现了端到端、bitwise batch-invariant 且确定性的 kernel，并且只带来最小的性能开销。这些 kernel 有助于调试、稳定性分析，以及保持一致的后训练行为。

**Batch Invariance。** Batch invariance 确保任意给定 token 的输出保持逐 bit 完全一致，而不受其在 batch 中位置的影响。为了实现 batch invariance，主要挑战如下：

* **Attention**。 为了实现 batch invariance，我们不能使用 split-KV 方法，该方法会将单个序列的注意力计算分布到多个 Stream Multiprocessors（SMs）上，以平衡 SM 的负载。然而，放弃该技术会导致严重的 wave-quantization 问题，从而对 GPU 利用率产生不利影响。为了解决这一问题，我们为 batch-invariant decoding 开发了一种双 kernel 策略。第一个 kernel 在单个 SM 内计算整个序列的注意力输出，从而在 wave 被完全占满时确保高吞吐量。第二个 kernel 为了最小化最后一个部分填充 wave 的延迟、从而缓解 wave-quantization，会为单个序列使用多个 SM。为了保证这两个 kernel 的逐 bit 一致性，我们仔细设计了第二个 kernel 的计算路径，以确保其累加顺序与第一个 kernel 相同。此外，第二个 kernel 在线程块集群内使用分布式共享内存，从而实现跨 SM 的高速数据交换。这种双 kernel 方法有效地将 batch-invariant decoding 的开销限制到可以忽略的程度。

* **Matrix Multiplication**。 传统的 cuBLAS 库无法实现 batch invariance。因此，我们使用 DeepGEMM 进行端到端替换。此外，对于非常小的 batch size，常规实现通常会采用 split-k 技术来提升性能。不幸的是，split-k 技术无法保证 batch invariance，而这是 DeepSeek-V4 中的一个关键特性。因此，我们在大多数场景下放弃 split-k，不过这可能会导致性能下降。为了解决这个问题，我们引入了一组优化，使我们的矩阵乘法实现能够在大多数主要场景中达到甚至超过标准 split-k 的性能。

**Determinism**。 确定性训练对于调试硬件或软件问题非常有帮助。此外，当训练中出现 loss spike 等异常时，确定性可以让研究人员更容易定位数值原因，并进一步改进模型设计。训练中的非确定性通常来自非确定性的累加顺序，这往往是由于使用 atomic addition 指令造成的。该问题主要发生在反向传播过程中，尤其是在以下部分：

* **Attention Backward**。 在传统的稀疏注意力反向传播实现中，我们使用 `atomicAdd` 来累加 KV token 的梯度。由于浮点加法不满足结合律，这会引入非确定性。为了解决这一问题，我们为每个 SM 分配独立的累加缓冲区，然后在所有缓冲区之间执行全局确定性求和。

* **MoE Backward**。 当来自不同 rank 的多个 SM 同时向接收 rank 上的同一缓冲区写入数据时，协商写入位置也会引入非确定性。为解决这一问题，我们在每个单独 rank 内设计了 token 顺序预处理机制，并结合跨多个 rank 的缓冲区隔离。该策略确保了专家并行发送结果的确定性，以及 MoE 反向传播中累加顺序的确定性。

* **Matrix Multiplication in mHC**。 mHC 涉及一次输出维度仅为 24 的矩阵乘法。对于非常小的 batch size，我们被迫使用 split-k 算法，而其朴素实现会导致非确定性。为克服这一点，我们分别输出每个 split 部分，并在随后的 kernel 中执行确定性归约，从而同时保留性能和确定性。


### 3.4 FP4 Quantization-Aware Training

### 3.5 Training Framework

我们的训练框架建立在为 DeepSeek-V3 开发的可扩展且高效的基础设施之上。在训练 DeepSeek-V4 时，我们继承了这一强大的基础，同时引入了几项关键创新，以适应其全新的架构组件——特别是 Muon 优化器、mHC 和混合注意力机制——并保持了较高的训练效率和稳定性。

#### 3.5.1 Efficient Implementation of Muon

#### 3.5.2 Cost-Effective and Memory-Efficient Implementation of mHC

#### 3.5.3 Contextual Parallelism for Long-Context Attention

#### 3.5.4 Extended Automatic Differentiation for Flexible Activation Checkpointing

### 3.6 Inference Framework

我们的推理框架很大程度上继承自 DeepSeek-V3，但在 KV 缓存管理方面有一些不同。

#### 3.6.1 KV Cache Structure and Management

#### 3.6.2 On-Disk KV Cache Storage

## 4.Pre-Training

### 4.1 Data Construction

在 DeepSeek-V3 的预训练数据基础上，我们致力于构建一个更加多样化、更高质量且包含更长有效上下文的训练语料库。我们不断优化数据构建流程。（1）对于**网络数据**，我们实施过滤策略，去除批量自动生成和模板化的内容，从而降低模型崩溃的风险。（2）**数学和编程语料库**仍然是我们训练数据的核心组成部分，我们在训练中期引入了智能体数据，进一步增强了 DeepSeek-V4 系列的编码能力。（3）对于**多语言数据**，我们为 DeepSeek-V4 构建了一个更大的语料库，提升了其在不同文化背景下对长尾知识的捕捉能力。（4）对于 DeepSeek-V4，我们特别注重**长文档数据**的整理，优先考虑科学论文、技术报告以及其他体现独特学术价值的材料。综上所述，我们的预训练语料库包含超过 32T 个 token，涵盖数学内容、代码、网页、长文档和其他高质量类别。

对于预训练数据，我们基本沿用了 DeepSeek-V3 的预处理策略。在分词方面，我们在 DeepSeek-V3 分词器的基础上，引入了一些用于构建上下文的特殊 token，并保持词表大小为 128K。我们也继承了 DeepSeek-V3 的 token 分割和中间填充（FIM）策略。受 Ding et al. (2024) 的启发，我们将来自不同来源的文档打包成合适的序列，以最大限度地减少样本截断。与 DeepSeek-V3 不同的是，我们在预训练过程中采用了**样本级注意力 mask**。

### 4.2 Pre-Training Setups

#### 4.2.1 Model Setups

**DeepSeek-V4-Flash**。我们将 Transformer 模型的层数设置为43，隐藏层维度 $d$ 设置为4096。前两层使用纯滑动窗口注意力机制。后续层则交替使用 CSA 和 HCA。对于 CSA，我们将压缩率 $m$ 设置为 4，索引器 query 头数量 $n^I_h$ 设置为 64，索引器头维度 $c^I$ 设置为 128，稀疏注意力（即注意力 top-k）选择的 KV 条目数设置为 512。对于 HCA，我们将压缩率 $m'$ 设置为 128。对于 CSA 和 HCA，我们将 query 头数量 $n_h$ 设置为 64，头维度 $c$ 设置为 512，query 压缩维度 $d_c$ 设置为 1024。输出投影组数量 $g$ 设置为 8，每个中间注意力输出的维度 $d_g$ 设置为 1024。对于滑动窗口注意力的附加分支，窗口大小 $n_{win}$ 设置为 128。在所有 Transformer 区块中使用 MoE 层，但前 3 个 MoE 层采用哈希路由策略。每个 MoE 层包含 1 个共享专家和 256 个路由专家，每个专家的中间隐藏维度为 2048。在这些路由专家中，每个 token 将激活 6 个专家。多 token 预测深度设置为 1。对于 mHC，扩展因子 $n_{hc}$ 设置为 4，Sinkhorn-Knopp 迭代次数 $t_{max}$ 设置为 20。在此配置下，DeepSeek-V4-Flash 总共包含 284B 参数，其中每个 token 激活 13B 参数。

**DeepSeek-V4-Pro**。我们将 Transformer 层数设置为 61，隐藏层维度 $d$ 设置为 7168。前两层使用 HCA 算法。后续层则交替使用 CSA 和 HCA 算法。对于 CSA，我们将压缩率 $m$ 设置为 4，索引器 query 头数量 $n^I_h$ 设置为 64，索引器头维度 $c_I$ 设置为 128，稀疏注意力（即注意力 top-k）选择的 KV 条目数设置为 1024。对于 HCA，我们将压缩率 $m'$ 设置为 128。对于 CSA 和 HCA，我们将 query 头数量 $n_h$ 设置为 128，头维度 $c$ 设置为 512，query 压缩维度 $d_c$ 设置为 1536。输出投影组数量 $g$ 设置为 16，每个中间注意力输出的维度 $d_g$ 设置为 1024。对于滑动窗口注意力的附加分支，窗口大小 $n_{win}$ 设置为128. 我们在所有 Transformer 区块中都采用了 MoE 层，但前 3 个 MoE 层使用哈希路由策略。每个 MoE 层包含 1 个共享专家和 384 个路由专家，每个专家的中间隐藏维度为 3072。在这些路由专家中，每个 token 将激活6个专家。多 token 预测深度设置为1。对于 mHC，扩展因子$n_{hc}$ 设置为4，Sinkhorn-Knopp迭代次数 $t_{max}$ 设置为20。在此配置下，DeepSeek-V4-Pro 总共包含 1.6T 参数，其中每个 token 激活 49B 参数。

#### 4.2.2 Training Setups

**DeepSeek-V4-Flash**。我们对大部分参数采用 Muon 优化器，但对嵌入模块、预测头模块以及所有 RMSNorm 模块的权重则采用 AdamW 优化器。对于 AdamW，我们将其超参数设置为 $β_1=0.9, β_2=0.95, ε=10^{−20}$ 和 weight_decay = 0.1。对于 Muon，我们将动量设置为 0.95，权重衰减设置为 0.1，并将每个更新矩阵的 RMS 重新缩放至 0.18，以便重复利用 AdamW 的学习率。我们使用 32T 个 token 训练 DeepSeek-V4-Flash 模型，并且与 DeepSeek-V3 类似，我们也采用了一种 **batch size 调度策略**，将 batch size（以 token 为单位）从较小的值逐渐增加到 75.5M，并在大部分训练过程中保持该值不变。在前 2000 步中，学习率线性提升，并在大部分训练过程中保持在 $2.7 × 10^{-4}$。接近训练结束时，我们最终按照余弦衰减策略将学习率衰减至 $2.7 × 10^{-5}$。训练从 4K 序列长度开始，然后逐步扩展到 16K、64K 和 1M。对于**稀疏注意力机制的设置**，我们首先使用密集注意力机制对前1T个 token 进行模型预热，然后在序列长度达到 64K 时引入稀疏注意力机制，并在剩余的训练过程中保持稀疏注意力机制。在引入注意力稀疏性时，我们首先设置一个短暂的阶段来预热 CSA 中的闪电索引器，然后在大部分训练过程中使用稀疏注意力机制训练模型。对于**辅助无损负载均衡**，我们将偏置更新速度设置为 0.001。为了避免单个序列内部出现极端的不平衡，我们将平衡损失的权重设置为 0.0001。**MTP 损失**的权重在大部分训练阶段设置为 0.3，在学习率衰减开始时设置为 0.1。

**DeepSeek-V4-Pro**。除了超参数的具体数值外，DeepSeek-V4-Pro 的训练设置与 DeepSeek-V4-Flash 基本一致。我们对大部分参数采用 Muon 优化器，但对嵌入模块、预测头模块以及所有 RMSNorm 模块的权重则采用 AdamW 优化器。AdamW 和 Muon 的超参数与 DeepSeek-V4-Flash 相同。我们使用 33T 个 token 训练 DeepSeek-V4-Pro，并采用**batch size 调度策略**，最大批大小为 94.4M 个 token。**学习率调度策略**与 DeepSeek-V4-Flash 基本相同，但峰值学习率设置为 $2.0 × 10^{-4}$，结束学习率设置为 $2.0 × 10^{-5}$。训练开始时序列长度为 4K，然后逐渐增加到 16K、64K 和 1M。与 DeepSeek-V4-Flash 相比，DeepSeek-V4-Pro 的密集注意力阶段持续时间更长，引入稀疏注意力的策略与 DeepSeek-V4-Flash 相同，均采用两阶段训练方法。为了实现**无损辅助负载均衡**，我们将偏置更新速度设置为 0.001。对于均衡损失，我们将其损失权重设置为 0.0001，以避免单个序列内出现极端不平衡。MTP 损失权重在大部分训练阶段设置为 0.3，在学习率衰减开始时设置为 0.1。

#### 4.2.3 Mitigating Training Instability

训练万亿参数的 MoE 模型面临着巨大的稳定性挑战，DeepSeek-V4 系列也不例外。我们在训练过程中遇到了显著的不稳定性问题。**虽然简单的回滚可以暂时恢复训练状态，但作为长期解决方案，它们并不理想，因为它们无法阻止损失尖峰的再次出现**。通过实验，我们发现尖峰的出现始终与 MoE 层中的异常值相关，而路由机制本身似乎会加剧这些异常值的出现。因此，我们尝试从两个方面解决这个问题：打破路由机制导致的恶性循环，以及直接抑制异常值。幸运的是，我们发现了两种能够有效维持训练稳定性的实用技术。尽管目前对其底层机制的全面理论理解仍有待完善，但我们公开分享这些技术，以促进社区的进一步探索。

**Anticipatory Routing**。我们发现，将骨干网络和路由网络的同步更新解耦可以显著提高训练稳定性。因此，在步骤 $t$，我们使用当前网络参数 $θ_t$ 进行特征计算，但路由索引则使用历史网络参数 $θ_{t−Δ_t}$ 进行计算和应用。实际上，为了避免重复加载模型参数带来的开销，我们在步骤 $t−Δt$ 预先获取步骤 $t$ 的数据。我们“预先”计算并缓存路由索引，以便在步骤 $t$ 中使用，因此我们将这种方法命名为“**预先路由**”。此外，我们还在基础设施层面对此进行了大量优化。首先，鉴于预计算路由索引仅需对数据进行一次前向传播，我们精心设计了流水线执行流程，并巧妙地将计算与专家并行（EP）通信相结合，成功地将预测路由的额外运行时间开销控制在约 20%。其次，我们引入了一种自动检测机制，当损失峰值出现时，该机制会触发短暂的回滚，并仅在需要时启用预测路由；在此模式下运行一段时间后，系统将恢复到标准训练模式。最终，这种动态应用使我们能够在几乎不增加额外训练开销的情况下避免损失峰值，且不会影响模型性能。

**SwiGLU Clamping**。在以往的文献中，clamping 已被明确用于约束数值范围，从而提高训练稳定性。在我们实际的训练过程中，我们通过经验发现，应用 SwiGLU clamping 可以有效地消除异常值，并显著有助于稳定训练过程，且不会影响性能。在 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的训练过程中，我们将 SwiGLU 的线性分量钳位在 [-10, 10] 的范围内，同时将门控分量的上限限制在 10。

### 4.3 Evaluations

#### 4.3.1 Evaluation Benchmarks

#### 4.3.2 Evaluation Results

## 5.Post-Training

### 5.1 Post-Training Pipeline

经过预训练后，我们进行了后训练阶段，最终得到了 DeepSeek-V4 系列模型。虽然训练流程与 DeepSeek-V3.2 基本相同，但关键的方法论上有所改动：混合强化学习 (RL) 阶段完全被 On-Policy Distillation (OPD) 所取代。

#### 5.1.1 Specialist Training

<img
  src="https://i-blog.csdnimg.cn/direct/1243aad6259b4f82a13b5bac8c4a1341.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

领域专家的开发是通过对 DeepSeek-V3.2 训练流程进行调整实现的。具体而言，每个模型都经过一系列优化步骤：首先进行初始微调，然后根据领域特定的提示和奖赏信号进行强化学习 (RL)。在强化学习阶段，我们采用了组相对策略优化 (GRPO) 算法，并保持超参数与我们之前的研究高度一致。

**Reasoning Efforts**。人们普遍认为，模型在推理任务上的性能从根本上取决于其计算量。因此，我们针对不同的强化学习配置训练了不同的专业模型，以促进针对不同推理能力进行优化的模型的开发。如表 2 所示，DeepSeek-V4-Pro 和 DeepSeek-V4-Flash 均支持三种特定的推理模式。对于每种模式，我们在强化学习训练期间应用不同的长度惩罚和上下文窗口，从而产生不同的推理输出 token 长度。为了整合这些不同的推理模式，我们使用了由 `<think>` 和`</think>` token 划分的专用响应格式。此外，对于“Think Max”模式，我们在 system prompt 的开头添加了一条特定的指令，以指导模型的推理过程，如表 3 所示。

**Generative Reward Model**。通常，易于验证的任务可以使用简单的基于规则的验证器或测试用例进行有效优化。相比之下，难以验证的任务传统上依赖于基于人类反馈的强化学习（RLHF），这需要大量的人工标注来训练标量奖赏模型。然而，在DeepSeek-V4系列的后训练阶段，我们摒弃了这些传统的基于标量的奖赏模型。相反，为了解决难以验证的任务，我们精心整理了基于评分标准的强化学习数据，并采用生成式奖赏模型（GRM）来评估策略轨迹。关键在于，我们将强化学习优化直接应用于 GRM 本身。在这种范式中，Actor 网络本身就充当 GRM，从而能够联合优化模型的评估（判断）能力及其标准的生成能力。通过统一这些角色，模型的内部推理能力被自然地融入到其评估过程中，从而产生高度鲁棒的评分。此外，该方法仅需极少的多样化人工标注即可取得优异的性能，因为该模型利用其自身的逻辑来泛化到复杂的任务中。

<img
  src="https://i-blog.csdnimg.cn/direct/725fdfca5a1748bd973bc03d2a522770.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Tool-Call Schema and Special Token**。与之前的版本一致，我们使用专用的 `<think></think>` 标签来描述推理路径。在 DeepSeek-V4 系列中，我们引入了一种新的工具调用模式，该模式采用特殊的 `|DSML|` token，并使用基于 XML 的工具调用格式，如表 4 所示。我们的实验表明，XML 格式有效地缓解了逃逸故障并减少了工具调用错误，从而为模型与工具的交互提供了更强大的接口。

<img
  src="https://i-blog.csdnimg.cn/direct/464526c1db734e35aa204069402d01c9.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Interleaved Thinking**。DeepSeek-V3.2 引入了一种上下文管理策略，该策略会在工具结果轮次之间保留推理轨迹，但在收到新的用户消息时将其丢弃。虽然这种策略有效，但在复杂的智能体工作流程中仍然会造成不必要的 token 浪费——每个新用户回合都会清除所有累积的推理内容，迫使模型从头开始重建其问题解决状态。利用扩展的 100 万 token 上下文，可以有效解决这个问题。
- **Tool-Calling Scenarios**。如图 7(a) 所示，整个对话过程中所有推理内容均被完整保留。与 DeepSeek-V3.2 在每个用户回合结束后丢弃思维轨迹不同，DeepSeek-V4 系列保留了所有回合的完整推理历史，包括跨越用户消息边界的推理。这使得模型能够在执行长期智能体任务时保持连贯且累积的思维链。
-  **General Conversational Scenarios**。如图 7(b) 所示，原始策略得以保留：当收到新的用户消息时，会丢弃先前回合的推理内容，从而在持久推理痕迹提供有限益处的场景中保持上下文简洁。

与 DeepSeek-V3.2 类似，通过用户消息模拟工具交互的 Agent 框架（例如 **Terminus**）可能不会触发工具调用上下文路径，因此可能无法受益于增强的推理持久性。我们仍然建议此类架构使用非思考模型。

**Quick Instruction**。在聊天机器人场景中，生成响应之前需要执行一些辅助任务（例如，判断是否触发网络搜索、意图识别等）。传统上，这些任务由一个独立的小型模型处理，由于**无法重用现有的 key-value 缓存，因此需要进行冗余的预填充**。为了克服这一限制，我们引入了快速指令（Quick Instruction）。我们将一组专用的特殊 token 直接附加到输入序列中，每个 token 对应一个特定的辅助任务。通过直接重用已计算的键值缓存，该机制完全避免了冗余的预填充，并允许并行执行某些任务，例如生成搜索查询以及确定权威性和域。因此，这种方法显著缩短了用户感知到的首次响应时间（TTFT），并消除了维护和迭代额外小型模型的工程开销。表 5 总结了支持的快速指令 token。

<img
  src="https://i-blog.csdnimg.cn/direct/f7d280f62aae4a3aa7108ce6be25460f.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

#### 5.1.2 On-Policy Distillation

在通过专门的微调和强化学习训练多个领域专家之后，我们采用**多教师  On-Policy Distillation (OPD)** 作为将专家能力融合到最终模型中的主要技术。OPD 已成为一种有效的后训练范式，能够高效地将领域专家的知识和能力迁移到单一的统一模型中。其实现方式是让 student 模型从 teacher 模型在其自身生成的轨迹上的输出分布中学习。形式上，给定一组 $N$ 个专家模型 $\{\pi_{E_1}, \pi_{E_2}, ..., \pi_{E_N}\}$，OPD 的目标函数定义如下：

```math
\mathcal L_{OPD}(\theta)=\sum^{N}_{i=1}w_i\cdot D_{KL}(\pi_{\theta}||\pi_{E_i}).\tag{29}
```

在此公式中，$w_i$ 表示分配给每个专家的权重，通常由专家的相对重要性决定。计算反向 KL 损失 $D_{KL}(\pi_{\theta}||\pi_{E_i})$ 需要从学生模型 $\pi_{\theta}$ 中采样训练轨迹，以保持 on-policy 学习。其底层逻辑确保统一策略 $pi_{\theta}$ 能够选择性地从与当前任务上下文相关的专家那里学习（例如，在数学推理任务中与数学专家对齐，在编程任务中与编码专家对齐）。通过这种机制，来自物理上不同的专家权重的知识通过 logits 级别的对齐整合到一个统一的参数空间中，从而有效地避免了传统权重合并或混合强化学习技术中经常遇到的性能下降问题。在此阶段，我们使用了十多个涵盖不同领域的 teacher 模型来蒸馏出一个 student 模型。

为了处理上述 OPD 目标，以往的工作通常将全词表 KL 损失简化为每个 token 位置的 token 级 KL 估计，并通过在策略损失计算中用 $sg\frac{log \pi_{E_i}(y_t|x,y_{<t})}{\pi_{\theta}(y_t|x,y_{<t})}$（sg 表示停止梯度操作）替换每个 token 的优势估计来复用强化学习框架。虽然这种方法资源效率高，但会导致梯度估计方差较大，并且常常造成训练不稳定。因此，我们在 OPD 中采用了全词表 logit 蒸馏。在计算反向 KL 损失时保留完整的 logit 分布可以得到更稳定的梯度估计，并确保教师知识的忠实蒸馏。在接下来的小节中，我们将描述使全词汇 OPD 能够大规模实现的工程工作。

### 5.2 RL and OPD Infrastructures

我们的后训练基础设施基于为 DeepSeekV3.2 开发的可扩展框架构建。具体而言，我们集成了第 3.5 节中描述的分布式训练堆栈以及之前介绍的用于高效自回归采样的 rollout 引擎。在此基础上，我们在本文中引入了以下主要改进。这些设计能够高效执行涉及十多个不同教师模型的超长上下文强化学习和 OPD 合并任务，从而显著加快模型发布的迭代周期。

#### 5.2.1 FP4 Quantization Integration

我们应用 FP4 (MXFP4) 量化来加速 rollout 和所有仅用于推理的前向传播，包括教师模型和参考模型的传播，从而减少内存流量和采样延迟。如第 3.4 节所述，我们在 rollout 和推理阶段直接使用原生 FP4 权重。对于训练步骤，我们通过无损的 FP4 到 FP8 反量化步骤来模拟 FP4 量化，从而可以无缝地重用现有的 FP8 混合精度框架和 FP32 主权重，而无需修改反向传播流水线。

#### 5.2.2 Efficient Teacher Scheduling for Full-Vocabulary OPD

我们的框架支持全词表 On-Policy Distillation (OPD)，且教师数量几乎不受限制，每个教师可能包含数万亿个参数。为了实现这一点，所有 teacher 权重都被卸载到集中式分布式存储中，并在 teacher 前向传播期间按需加载，采用类似 ZeRO 的参数分片技术来缓解 I/O 和 DRAM 压力。此外，即使将 logits 写入磁盘，对于词表 $|V| > 100k$ 的所有 teacher，直接实现 logits 的计算量也相当巨大。我们通过在前向传播期间仅将最后一层 teacher 的隐藏状态缓存到集中式缓冲区来解决这个问题。在训练时，这些缓存状态会被检索并传递给相应的预测头模块，以动态重建完整的 logits。这种设计产生的重新计算开销可以忽略不计，同时完全避免了显式实现 logits 所带来的内存负担。为了减少 teacher 预测头的 GPU 内存占用，我们在数据分发期间按 teacher 索引对训练样本进行排序。这种安排确保每个不同的 teacher 模型头在每个小批次中只加载一次，并且任何给定时间设备内存中最多只驻留一个 teacher 模型头。所有参数和隐藏状态的加载/卸载操作都在后台异步进行，不会阻塞关键路径上的计算。最后，teacher 和 student logits之间的精确 KL 散度使用专门的TileLang内核计算，这可以加速计算并减少动态内存分配。

#### 5.2.3 Preemptible and Fault-Tolerant Rollout Service

为了最大化 GPU 资源利用率，同时支持为高优先级任务快速调配硬件，我们的 GPU 集群采用了集群级的抢占式任务调度器，其中任何正在运行的任务都可能在任意时刻被抢占。此外，在大规模 GPU 集群中，硬件故障也十分常见。为此，我们为 RL/OPD rollout 实现了一种可抢占且容错的 LLM 生成服务。

具体来说，我们为每个生成请求实现了 token 粒度的 Write-Ahead Log（WAL，预写日志）。每当某个请求生成一个新的 token，我们都会立即将其追加到该请求的 WAL 中。在发生抢占时，我们暂停推理引擎，并保存未完成请求的 KV cache。恢复时，我们使用持久化的 WAL 和已保存的 KV cache 继续解码。即使发生致命硬件错误，我们也可以使用 WAL 中持久化保存的 token 重新运行 prefill 阶段，以重建 KV cache。

重要的是，从数学上讲，从头重新生成未完成请求是不正确的，因为这会引入长度偏差。由于较短的响应更有可能在中断中幸存下来，因此每当发生中断时，从头重新生成会使模型更倾向于产生较短序列。如果推理栈具备 batch-invariant 和确定性特性，那么这个正确性问题也可以通过使用采样器中伪随机数生成器的一致 seed 重新生成来解决。然而，这种方法仍然会带来重新运行解码阶段的额外成本，因此远不如我们的 token 粒度 WAL 方法高效。

#### 5.2.4 Scaling RL Framework for Million-Token Context

我们为百万 token 序列上的高效 RL 和 OPD 引入了针对性优化。在 rollout 阶段，我们采用了可抢占且容错的 rollout 服务，详见第 5.2.3 节。对于推理和训练阶段，我们将 rollout 数据格式分解为轻量级元数据和较重的逐 token 字段。在数据分发过程中，可以加载整个 rollout 数据的元数据，用于执行全局 shuffle 和 packing layout 计算。较重的逐 token 字段则通过共享内存数据加载器加载，以消除节点内数据冗余，并在 mini-batch 粒度消费后立即释放，从而显著降低 CPU 和 GPU 的内存压力。设备上的 mini-batch 数量会根据工作负载动态确定，从而在计算吞吐量和 I/O 重叠之间实现高效权衡。

#### 5.2.5  Sandbox Infrastructure for Agentic AI

为了满足 agentic AI 在后训练和评估阶段多样化的执行需求，我们构建了一个生产级沙箱平台 **DeepSeek Elastic Compute（DSec）**。DSec 由三个 Rust 组件组成——API gateway（`Apiserver`）、每台 host 上的 agent（`Edge`）以及集群监控器（`Watcher`）——它们通过自定义 RPC 协议互连，并基于 3FS 分布式文件系统（DeepSeek-AI, 2025）进行水平扩展。在生产环境中，单个 DSec 集群可以管理数十万个并发沙箱实例。

DSec 的设计受到四点观察的驱动：（1）agentic 工作负载高度异构，范围从轻量级函数调用到完整的软件工程流水线，并具有多样化的操作系统和安全需求；（2）环境镜像数量众多且体积庞大，但必须能够快速加载并支持迭代式定制；（3）高密度部署要求高效利用 CPU 和内存；（4）沙箱生命周期必须与 GPU 训练调度协同，包括抢占和基于 checkpoint 的恢复。基于这些观察，我们在下文分别阐述 DSec 的四项核心设计。

**一个统一接口背后的四种执行基底。** DSec 暴露了一个单一的 Python SDK（`libdsec`），用于抽象四种执行基底。**Function Call** 将无状态调用分发到预热的容器池，从而消除冷启动开销。**Container** 完全兼容 Docker，并利用 EROFS（Gao et al., 2019）的按需加载来实现高效镜像组装。**microVM** 基于 Firecracker（Agache et al., 2020）构建，为安全敏感、高密度部署提供 VM 级隔离。**fullVM** 基于 QEMU（Bellard, 2005）构建，支持任意 guest 操作系统。四者共享一个通用 API 表面——命令执行、文件传输和 TTY 访问——在它们之间切换只需要修改一个参数。

**通过分层存储实现快速镜像加载。** DSec 通过分层、按需加载，在快速启动与庞大且持续增长的环境镜像语料之间取得平衡。对于容器，基础镜像和文件系统提交会存储为由 3FS 支持的只读 EROFS 层，并直接挂载到 overlay 的 `lowerdirs` 中。我们在挂载时将文件元数据保持在本地磁盘上随时可用；同时，数据块会根据请求从 3FS 拉取。对于 microVM，DSec 使用 `overlaybd`（Li et al., 2020）磁盘格式：只读基础层位于 3FS 上以支持跨实例共享，而写入会进入本地 copy-on-write 层。这类快照可以串联起来，从而支持高效版本管理和毫秒级恢复。

**大规模并发下的密度优化。** 为了支持每个集群数十万个沙箱，DSec 解决了两个资源瓶颈。首先，它缓解了虚拟化环境中重复 page-cache 占用的问题，并应用内存回收以支持安全的超额分配。其次，它缓解了容器运行时中的 spinlock 竞争，因此降低了每个沙箱的 CPU 开销，显著提升了每台 host 的装箱密度。

**轨迹日志与抢占安全恢复。** DSec 为每个沙箱维护一个全局有序的轨迹日志，持久化记录每一次命令调用及其结果。该轨迹有三个用途：（1）**client fast-forwarding**——当训练任务被抢占时，沙箱资源仍会被保留；恢复时，DSec 会为先前已完成的命令重放缓存结果，从而加速任务恢复，同时避免重新执行非幂等操作所导致的错误；（2）**细粒度溯源**——每个状态变化的来源及其对应结果都是可追踪的；（3）**确定性重放**——任何历史会话都可以根据其轨迹被忠实复现。

### 5.3 Standard Benchmark Evaluation
