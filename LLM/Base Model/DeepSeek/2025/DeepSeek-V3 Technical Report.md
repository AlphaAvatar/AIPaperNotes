
论文链接：https://arxiv.org/pdf/2412.19437

代码链接：

# 摘要

我们提出了 DeepSeek-V3，这是一个强大的混合专家 (MoE) 语言模型，总共有 671B 个参数，每个 token 激活 37B。为了实现高效的推理和经济高效的训练，DeepSeek-V3 采用了多头潜在注意力 (MLA) 和 DeepSeekMoE 架构，这些架构在 DeepSeek-V2 中得到了彻底的验证。此外，DeepSeek-V3 开创了一种无辅助损失的负载平衡策略，并设置了多 token 预测训练目标以获得更强大的性能。我们在 14.8 万亿个多样化和高质量的 token 上对 DeepSeek-V3 进行了预训练，然后进行有监督微调和强化学习阶段，以充分利用其功能。综合评估表明，DeepSeek-V3 优于其他开源模型，并实现了与领先的闭源模型相当的性能。尽管性能出色，但 DeepSeek-V3 仅需要 2.788M H800 GPU 小时即可完成完整训练。此外，它的训练过程非常稳定。在整个训练过程中，我们没有遇到任何不可恢复的损失峰值或执行任何回滚。模型检查点可在 https://github.com/deepseek-ai/DeepSeek-V3 上找到。

# 1.Introduction
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e269e5087cd0460996b2a21624a06355.png)

近年来，大型语言模型 (LLM) 经历了快速迭代和演进，与通用人工智能 (AGI) 的差距正在逐步缩小。除了闭源模型外，开源模型，包括 DeepSeek 系列、LLaMA 系列、Qwen 系列和 Mistral 系列也在取得重大进展，努力缩小与闭源模型的差距。为了进一步突破开源模型能力的界限，我们扩大了模型规模，并推出了 DeepSeek-V3，这是一个大型混合专家 (MoE) 模型，具有 671B 个参数，其中每个 token 激活 37B 个参数。

从未来的角度看，我们始终追求模型性能强劲、成本低廉。因此，在架构方面，DeepSeek-V3 仍然采用多头潜在注意力 (MLA) 实现高效推理，采用 DeepSeekMoE 实现低成本训练。这两种架构已在 DeepSeekV2 中得到验证，证明了它们能够在实现高效训练和推理的同时保持稳健的模型性能。除了基本架构之外，我们还实现了两种额外策略来进一步增强模型能力。首先，DeepSeek-V3 率先采用无辅助损失策略来实现负载平衡，旨在最大限度地减少因鼓励负载平衡而对模型性能产生的不利影响。其次，DeepSeek-V3 采用多 token 预测训练目标，我们观察到这可以提高评估基准上的整体性能。

为了实现高效训练，我们支持 FP8 混合精度训练，并对训练框架进行了全面的优化。低精度训练已经成为一种很有前途的高效训练解决方案，它的演进与硬件能力的进步密切相关。在这项工作中，我们引入了一个 FP8 混合精度训练框架，并首次在超大规模模型上验证了其有效性。通过对 FP8 计算和存储的支持，我们实现了加速训练和减少GPU内存使用。至于训练框架，我们设计了 DualPipe 算法来实现高效的流水线并行，它具有更少的流水线气泡，并通过计算-通信重叠隐藏了训练过程中的大部分通信。这种重叠确保了随着模型进一步扩大，只要我们保持恒定的计算与通信比率，我们仍然可以在节点之间使用细粒度的专家，同时实现接近于零的全对全通信开销。此外，我们还开发了高效的跨节点全对全通信内核，以充分利用InfiniBand（IB）和NVLink带宽。此外，我们还精心优化了内存占用，使得无需使用昂贵的张量并行即可训练 DeepSeek-V3。结合这些努力，我们实现了较高的训练效率。

在预训练阶段，我们基于 14.8T 高质量、多样化的 token 对 DeepSeek-V3 进行训练。预训练过程非常稳定，整个训练过程中没有出现不可挽回的 loss spike 和回滚的情况。接下来，我们对 DeepSeek-V3 进行了两阶段的上下文长度扩展，第一阶段将最大上下文长度扩展到 32K，第二阶段进一步扩展到 128K。之后，我们在 DeepSeek-V3 的基础模型上进行包括有监督微调 (SFT) 和强化学习 (RL) 在内的后训练，使其更贴近人类的偏好，进一步释放其潜力。在后训练阶段，我们从 DeepSeek-R1 系列模型中提炼推理能力，同时小心地保持模型准确率和生成长度之间的平衡。

我们根据一系列全面的基准测试对 DeepSeek-V3 进行了评估。尽管训练成本不高，但全面的评估表明，DeepSeek-V3-Base 已成为目前最强大的开源基础模型，尤其是在代码和数学方面。其chat版本也优于其他开源模型，并且在一系列标准和开放式基准测试中实现了与领先的闭源模型（包括 GPT-4o 和 Claude-3.5-Sonnet）相当的性能。

最后，我们再次强调 DeepSeek-V3 的经济训练成本，如表 1 所示，这是通过我们优化算法、框架和硬件的协同设计实现的。在预训练阶段，每万亿个 token 上训练 DeepSeek-V3 仅需要 180K H800 GPU 小时，也就是说，在我们拥有 2048 个 H800 GPU 的集群上需要 3.7 天。因此，我们的预训练阶段在不到两个月的时间内完成，花费了 2664K GPU 小时。加上上下文长度扩展的 119K GPU 小时和后训练的 5K GPU 小时，DeepSeek-V3 的完整训练仅花费 278.8 万 GPU 小时。假设 H800 GPU 的租赁价格为每 GPU 小时 2 美元，我们的总训练成本仅为 557.6 万美元。请注意，上述成本仅包括 DeepSeek-V3 的官方训练，不包括与架构、算法或数据的先前研究和消融实验相关的成本。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fb16d1e60c4749ca803fd1beafd6e19a.png)

我们的主要贡献包括:

**Architecture: Innovative Load Balancing Strategy and Training Objective。**
- 在 DeepSeek-V2 高效的架构之上，我们首创了一种无辅助损失的负载平衡策略，最大限度地减少了因鼓励负载平衡而导致的性能下降。
- 我们研究了多 token 预测 (MTP) 目标，并证明它有利于模型性能。它还可以用于推测解码以加速推理。

**Pre-Training: Towards Ultimate Training Efficiency**。
- 我们设计了FP8混合精度训练框架，并首次在超大规模模型上验证了FP8训练的可行性和有效性。
- 通过算法、框架和硬件的协同设计，我们克服了跨节点 MoE 训练中的通信瓶颈，实现了近乎完全的计算通信重叠。这大大提高了我们的训练效率并降低了训练成本，使我们能够在不增加额外开销的情况下进一步扩大模型规模。
- 我们以仅 2.664M H800 GPU 小时的经济成本，在 14.8T token 上完成了 DeepSeek-V3 的预训练，得到了目前最强的开源基础模型，预训练之后的后续训练阶段仅需 0.1M GPU 小时。

**Post-Training: Knowledge Distillation from DeepSeek-R1**。
- 我们引入了一种创新方法，将长思维链 (CoT) 模型（特别是 DeepSeek R1 系列模型之一）中的推理能力提炼到标准 LLM（尤其是 DeepSeek-V3）中。我们的流程将 R1 的验证和反思模式巧妙地融入到 DeepSeek-V3 中，并显著提高了其推理性能。同时，我们还保持对 DeepSeek-V3 的输出样式和长度的控制。

**Summary of Core Evaluation Results**。
- **Knowledge**:（1）在MMLU、MMLU-Pro、GPQA等教育类Benchmark上，DeepSeek-V3的表现均优于其他开源模型，MMLU得分为88.5，MMLU-Pro得分为75.9，GPQA得分为59.1，性能可与GPT-4o、Claude-Sonnet-3.5等领先的闭源模型相媲美，缩小了该领域开源与闭源模型之间的差距。（2）在事实​​性Benchmark上，DeepSeek-V3在SimpleQA和中文SimpleQA上均表现出优于开源模型的性能。虽然它在英文事实性知识（SimpleQA）上落后于GPT-4o和Claude-Sonnet-3.5，但在中文事实性知识（中文SimpleQA）上却超越了这些模型，凸显了其在中文事实性知识方面的实力。
- **Code, Math, and Reasoning**：（1）在所有 non-long-CoT 开源和闭源模型中，DeepSeek-V3 在数学相关基准测试中取得了最佳性能。值得注意的是，它甚至在 MATH-500 等特定基准测试中超越了 o1-preview，展示了其强大的数学推理能力。（2）在编码相关任务中，DeepSeek-V3 成为编码竞赛基准测试（如 LiveCodeBench）中表现最好的模型，巩固了其作为该领域领先模型的地位。对于工程相关任务，虽然 DeepSeek-V3 的表现略低于 Claude-Sonnet-3.5，但它仍然以显著的优势领先于所有其他模型，展示了其在各种技术基准测试中的竞争力。

在本文的其余部分，我们首先详细介绍了我们的 DeepSeek-V3 模型架构（第 2 节）。随后，我们介绍我们的基础设施，包括我们的计算集群、训练框架、对 FP8 训练的支持、推理部署策略以及我们对未来硬件设计的建议。接下来，我们描述我们的预训练过程，包括训练数据的构建、超参数设置、长上下文扩展技术、相关评估以及一些讨论（第 4 节）。此后，我们讨论了我们在训练后方面的努力，其中包括有监督微调 (SFT)、强化学习 (RL)、相应的评估和讨论（第 5 节）。最后，我们总结这项工作，讨论 DeepSeek-V3 的现有局限性，并提出未来研究的潜在方向（第 6 节）。

# 2.Architecture

我们首先介绍 DeepSeek-V3 的基本架构，其特点是多头潜在注意力 (MLA) 可实现高效推理，DeepSeekMoE 可实现经济的训练。然后，我们提出多 token 预测 (MTP) 训练目标，我们观察到该目标可提高评估基准上的整体性能。对于未明确提及的其他细节，DeepSeek-V3 遵循 DeepSeekV2 的设置。

## 2.1 Basic Architecture
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce0aba2388de427eb4987ec54f5b26e6.png)

DeepSeek-V3 的基本架构仍然在 Transformer 框架内。为了实现高效的推理和经济的训练，DeepSeek-V3 还采用了 MLA 和 DeepSeekMoE，这些技术在 DeepSeek-V2 中得到了充分的验证。与 DeepSeek-V2 相比，一个例外是，我们为 DeepSeekMoE 额外引入了一种无辅助损失的负载平衡策略，以减轻为保证负载平衡而导致的性能下降。图 2 说明了 DeepSeek-V3 的基本架构，我们将在本节中简要回顾 MLA 和 DeepSeekMoE 的细节。

### 2.1.1. Multi-Head Latent Attention

对于注意力机制，DeepSeek-V3 采用了 MLA 架构。设 $d$ 表示嵌入维度，$n_h$ 表示注意力头的数量，$d_h$ 表示每个注意力头的维度，$\textbf h_t ∈ \mathbb R^d$ 表示给定注意力层中第 $t$ 个 token 的注意力输入。MLA 的核心是对注意力 key 和 value 进行低秩联合压缩，以减少推理过程中的键值 (KV) 缓存：

$$\begin{array}{cc}
c^{KV}_t=W^{DKV}\textbf h_t, & (1)\\
[\textbf k^C_{t,1};\textbf k^C_{t,2};...;\textbf k^C_{t,n_h}]=\textbf k^C_t=W^{UK}\textbf c^{KV}_t, & (2)\\
\textbf k^R_t=RoPE(W^{KR}\textbf h_t), & (3)\\
\textbf k_{t,i}=[\textbf k^C_{t,i};\textbf k^R_t], & (4)\\
[\textbf v^C_{t,1};\textbf v^{C}_{t,2};...;\textbf v^{C}_{t,n_h}]=\textbf v^C_t = W^{UV}c^{KV}_t, & (5)
\end{array}$$

其中 $\textbf c^{KV}_t ∈ \mathbb R^{d_c}$ 是 key 和 value 的潜在压缩向量；$d_c(\ll d_hn_h)$ 表示 KV 压缩维度；$W^{DKV} ∈ \mathbb R^{d_c×d}$ 表示下投影矩阵；$W^{UK},W^{UV} ∈ \mathbb R^{d_hn_h×d_c}$ 分别是 key 和 value 的上投影矩阵；$W^{KR} ∈ \mathbb R^{d^R_h×d}$ 是用于生成携带旋转位置嵌入 (RoPE) 的解耦 key 的矩阵；RoPE(·) 表示应用 RoPE 矩阵的操作；[·; ·] 表示拼接。请注意，对于 MLA，在生成过程中只需要缓存蓝框向量（即 $c^{KV}_t$ 和 $k^R_t$），这会显著减少 KV 缓存，同时保持与标准多头注意力 (MHA) 相当的性能。

对于注意力 query，我们还执行低秩压缩，这可以减少训练期间的激活记忆：

$$\begin{array}{cc}
\textbf c^Q_t=W^{DQ}\textbf h_t, & (6)\\
[\textbf q^C_{t,1};\textbf q^C_{t,2};...;\textbf q^C_{t,n_h}]=\textbf q^C_t=W^{UQ}\textbf c^Q_t, & (7)\\
[\textbf q^R_{t,1};\textbf q^R_{t,2};...;\textbf q^R_{t,n_h}]=RoPE(W^{QR}\textbf c^Q_t), & (8)\\
\textbf q_{t,i}=[\textbf q^C_{t,i};\textbf q^R_{t,i}], & (9)
\end{array}$$

其中 $\textbf c^Q_t ∈ \mathbb R^{d'_c}$ 是 query 的潜在压缩向量；$d'_c(\ll d_hn_h)$ 表示 query 的压缩维度；$\textbf W^{DQ} ∈ \mathbb R^{d'_c×d}, W^{UQ}∈ \mathbb R^{d_hn_h×d'_c}$ 分别是 query 的下投影矩阵和上投影矩阵；$W^{QR} ∈ \mathbb R^{d^Rn_h×d'c}$ 是生成携带 RoPE 的解耦 query 的矩阵。

最终，注意力 query ($\textbf q_{t,i}$)、key ($\textbf k_{j,i}$) 和 value ($\textbf v^C_{j,i}$) 组合起来，得到最终的注意力输出 $u_t$：

$$\begin{array}{cc}
\textbf o_{t,i} = \sum^t_{j=1}Softmax_j(\frac{\textbf q^T_{t,i}\textbf k_{j,i}}{\sqrt{d_h+d^R_h}})\textbf v^C_{j,i}， &  (10)
\textbf u_t=W^O[\textbf o_{t,1};\textbf o_{t,2};...;\textbf o_{t,n_h}], & (11)
\end{array}$$

其中$W^O\in\mathbb R^{d\times d_hn_h}$表示输出投影矩阵。
### 2.1.2. DeepSeekMoE with Auxiliary-Loss-Free Load Balancing

**Basic Architecture of DeepSeekMoE**。对于前馈网络 (FFN)，DeepSeek-V3 采用 DeepSeekMoE 架构。与 GShard 等传统 MoE 架构相比，DeepSeekMoE 使用更细粒度的专家，并将一些专家隔离为共享专家。设 $\textbf u_t$ 表示第 $t$ 个 token 的 FFN 输入，我们按如下方式计算 FFN 输出 $\textbf h'_t$：

$$\begin{array}{cc}
\textbf h'_t=\textbf u_t+\sum^{N_s}_{i=1}FFN^{(s)}_i(\textbf u_t)+\sum^{N_r}_{i=1}g_{i,t}FFN^{(r)}_t(\textbf u_t), & (12)\\
g_{i,t}=\frac{g'_{i,t}}{\sum^{N_r}_{j=1}g'_{j,t}}, & (13)\\
g'_{i,t}=\begin{cases}
s_{i,t}, & s_{i,t} \in Topk(\{s_{j,t}|1\le j\le N_r\},K_r),\\
0, & otherwise,
\end{cases} & (14)\\
s_{i,t}=Sigmoid(\textbf u^T_t\textbf e_i), & (15)
\end{array}$$

其中 $N_s$ 和 $N_r$ 分别表示共享专家和路由专家的数量；$FFN^{(s)}_i(·)$ 和 $FFN^{(r)}_i(·)$ 分别表示第 $i$ 个共享专家和第 $i$ 个路由专家；$K_r$ 表示激活的路由专家的数量；$g_{i,t}$ 是第 $i$ 个专家的门控值；$s_{i,t}$ 是 token 到专家的亲和力；$\textbf e_i$ 是第 $i$ 个路由专家的质心向量；$Topk(·,K)$ 表示在为第 $t$ 个 token 和所有路由专家计算的亲和力分数中 $K$ 个最高分数的集合。与 DeepSeek-V2 略有不同，DeepSeek-V3 使用 Sigmoid 函数计算亲和力分数，并在所有选定的亲和力分数之间应用标准化来产生门控值。

**Auxiliary-Loss-Free Load Balancing**。对于 MoE 模型，各专家间负载不均衡会导致路由崩溃，并在专家并行场景中降低计算效率。传统解决方案通常依靠辅助损失来避免负载不均衡。然而，过大的辅助损失会损害模型性能。为了在负载平衡和模型性能之间取得更好的平衡，我们首创了一种无辅助损失的负载均衡策略来确保均载平衡。具体来说，我们为每个专家引入一个偏差项 $b_i$，并将其添加到相应的亲和力分数 $s_{i,t}$ 中以确定 top-K 路由：

$$g'_{i,t}=\begin{cases}
s_{i,t}, & s_{i,t}+b_i \in Topk(\{s_{j,t}+b_j|1\le j\le N_r\},K_r),\\
0, & otherwise.
\end{cases}\tag{16}$$

请注意，偏差项仅用于路由。将与 FFN 输出相乘的门控值仍来自原始亲和力得分 $s_{i,t}$。在训练期间，我们会持续监控每个训练步骤的整个批次上的专家负载。在每个步骤结束时，如果其对应的专家超载，我们将偏差项减少 $\gamma$，如果其对应的专家负载不足，我们将偏差项增加 $\gamma$，其中 $\gamma$ 是一个称为偏差更新速度的超参数。通过动态调整，DeepSeek-V3 在训练期间保持专家负载平衡，并且比通过纯辅助损失来鼓励负载平衡的模型获得更好的性能。

**Complementary Sequence-Wise Auxiliary Loss**。虽然 DeepSeek-V3 主要依靠无辅助损失策略实现负载平衡，但为了防止任何单个序列内的极端不平衡，我们还采用了互补的序列平衡损失：

$$\mathcal L_{Bal}=\alpha\sum^{N_r}_{i=1}f_iP_i,\tag{17}$$
$$f_i=\frac{N_r}{K_rT}\sum^T_{t=1}\mathbb I(s_{i,t}\in Topk(\{s_{j,t}|1\le j\le N_r\},K_r)),\tag{18}$$
$$s'_{i,t}=\frac{s_{i,t}}{\sum^{N_r}_{j=1}s_{j,t}},\tag{19}$$
$$P_i=\frac{1}{T}\sum^T_{t=1}s'_{i,t},\tag{20}$$

其中平衡因子 $\alpha$ 是一个超参数，对于 DeepSeek-V3，它将被分配一个非常小的值；$\mathbb I(·)$ 表示指示函数；$T$ 表示序列中的 token 数。序列平衡损失鼓励每个序列上的专家负载保持平衡。

**Node-Limited Routing**。与 DeepSeek-V2 使用的 device-limited 路由一样，DeepSeek-V3 也使用受限路由机制来限制训练期间的通信成本。简而言之，我们确保每个 token 最多被发送到 $M$ 个节点，这些节点是根据每个节点上分布的最高 $\frac{K_r}{M}$ 个专家的亲和力得分的总和来选择的。在此约束下，我们的 MoE 训练框架几乎可以实现完全的计算-通信重叠。

**No Token-Dropping**。由于采用了有效的负载均衡策略，DeepSeek-V3 在整个训练过程中保持了良好的负载均衡，因此 DeepSeek-V3 在训练过程中不会丢掉任何 token。此外，我们还实现了特定的部署策略来确保推理负载均衡，因此 DeepSeek-V3 在推理过程中也不会丢掉 token。

## 2.2 Multi-Token Prediction
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/48c7280c7c084ce29470e87209dbc422.png)

受到 Gloeckle et al. (2024) 的启发，我们研究并为 DeepSeek-V3 设置了一个多 token 预测 (MTP) 目标，该目标将预测范围扩展到每个位置的多个未来 token。一方面，MTP 目标使训练信号密集化并可能提高数据效率。另一方面，MTP 可能使模型能够预先规划其表示，以更好地预测未来的token。图 3 说明了我们对 MTP 的实现。与 Gloeckle et al. (2024) 使用独立输出头并行预测 𝐷 个额外 token 不同，我们按顺序预测其他 token 并在每个预测深度保留完整的因果链。我们将在本节中介绍 MTP 实现的细节。

**MTP Modules**。具体来说，我们的 MTP 实现使用 𝐷 个序列模块来预测 $D$ 个额外token。第 $k$ 个 MTP 模块由共享嵌入层 $Emb(·)$、共享输出头 $OutHead(·)$、Transformer 块 $TRM_k(·)$ 和投影矩阵 $M_k ∈ \mathbb R^{d×2d}$ 组成。对于第 $i$ 个输入 token $t_i$，在第 $k$ 个预测深度，我们首先将第 $(k − 1)$ 个深度处第 $i$ 个 token 的表示 $\textbf h^{k−1}_i∈ \mathbb R^d$ 和第 $(i+k)$ 个token 的嵌入$Emb(t_{i+k}) ∈ \mathbb R^d$结合，然后进行线性投影：

$$\textbf h'^k_i=M_k[RMSNorm(\textbf h^{k-1}_i);RMSNorm(Emb(t_{i+k}))],\tag{21}$$

其中 $[·; ·]$ 表示拼接。特别是，当 $k = 1$ 时，$\textbf h^{k−1}_i$ 指的是主模型给出的表示。请注意，对于每个 MTP 模块，其嵌入层与主模型共享。组合的 $\textbf h'^k_i$ 用作第 $k$ 个深度的 Transformer 块的输入，以产生当前深度的输出表示$\textbf h^k_i$：

$$\textbf h^k_{1:T-k}=TRM_k(\textbf h'^k_{1:T-k}),\tag{22}$$

其中 $T$ 表示输入序列长度，$i:j$ 表示切片操作（包括左边界和右边界）。最后，以 $\textbf h^k_i$ 作为输入，共享输出头将计算第 $k$ 个附加预测 token 的概率分布 $P^k_{i+1+k}∈\mathbb R^V$，其中 $V$ 是词表大小：

$$p^k_{i+k+1}=OutHead(\textbf h^k_i).\tag{23}$$

输出头 $OutHead(·)$ 将表示线性映射到 logits，随后应用 $Softmax(·)$ 函数来计算第 $k$ 个附加token的预测概率。此外，对于每个 MTP 模块，其输出头与主模型共享。我们维护预测因果链的原理与 EAGLE 类似，但其主要目标是推测解码，而我们利用 MTP 来改进训练。

**MTP Training Objective**。对于每个预测深度，我们计算交叉熵损失 $\mathcal L_^k_{MTP}$：

$$\mathcal L^k_{MTP}=CrossEntropy(p^k_{s+k:T+1},t_{2+k:T+1})=-frac{1}{T}\sum^{T+1}_{i=2+k}log~p^k_i[t_i],\tag{24}$$

其中 $T$ 表示输入序列长度，$t_i$ 表示第 $i$ 个位置的真实token，$p^k_i[t_i]$ 表示第 $k$ 个 MTP 模块给出的 $t_i$ 的相应预测概率。最后，我们计算所有深度的 MTP 损失的平均值，并将其乘以加权因子 $\lambda$ 以获得整体 MTP 损失 $\mathcal L_{MTP}$，这可作为 DeepSeek-V3 的额外训练目标：

$$\mathcal L_{MTP}=\frac{\lambda}{D}\sum^D_{k=1}\mathcal L^k_{MTP}.\tag{25}$$

**MTP in Inference**。我们的 MTP 策略主要是为了提升主模型的性能，因此在推理过程中，我们可以直接丢弃 MTP 模块，主模型可以独立正常运行。此外，我们还可以重新利用这些 MTP 模块进行推测解码，以进一步改善生成延迟。
# 3.Infrastructures
## 3.1 Compute Clusters

DeepSeek-V3 在配备 2048 个 NVIDIA H800 GPU 的集群上进行训练。H800 集群中的每个节点包含 8 个 GPU，通过节点内的 NVLink 和 NVSwitch 连接。在不同的节点之间，使用 InfiniBand (IB) 互连来促进通信。

## 3.2 Training Framework

DeepSeek-V3 的训练由 HAI-LLM 框架支持，这是一个高效且轻量级的训练框架，由我们的工程师从头开始打造。总体而言，DeepSeek-V3 应用了 16 路流水线并行 (PP)、跨越 8 个节点的 64 路专家并行 (EP) 和 ZeRO-1 数据并行 (DP)。

为了提高 DeepSeek-V3 的训练效率，我们进行了细致的工程优化。首先，我们设计了 DualPipe 算法以实现高效的流水线并行。与现有的 PP 方法相比，DualPipe 具有更少的流水线气泡。更重要的是，它将计算和通信阶段重叠在前向和后向过程中，从而解决了跨节点专家并行引入的沉重通信开销的挑战。其次，我们开发了高效的跨节点全对全通信内核，以充分利用 IB 和 NVLink 带宽并节省专用于通信的流多处理器 (SM)。最后，我们精心优化了训练期间的内存占用，从而使我们能够在不使用昂贵的张量并行 (TP) 的情况下训练 DeepSeek-V3。

### 3.2.1. DualPipe and Computation-Communication Overlap
### 3.2.2. Efficient Implementation of Cross-Node All-to-All Communication
### 3.2.3. Extremely Memory Saving with Minimal Overhead
## 3.3. FP8 Training

受低精度训练最新进展的启发，我们提出了一种细粒度混合精度框架，利用 FP8 数据格式来训练 DeepSeek-V3。虽然低精度训练前景广阔，但它通常受到激活、权重和梯度中存在异常值的限制。尽管在推理量化方面取得了重大进展，但很少有研究证明低精度技术在大规模语言模型预训练中的成功应用。为了应对这一挑战并有效扩展 FP8 格式的动态范围，我们引入了一种细粒度量化策略：使用 $1 × N_c$ 元素进行逐块分组或使用 $N_c × N_c$ 元素进行逐块分组。在我们提高精度的累积过程中，相关的反量化开销得到了很大程度的减轻，这是实现精确的 FP8 通用矩阵乘法 (GEMM) 的关键方面。此外，为了进一步减少 MoE 训练中的内存和通信开销，我们在 FP8 中缓存和调度激活，同时在 BF16 中存储低精度优化器状态。我们在两个类似于 DeepSeek-V2-Lite 和 DeepSeekV2 的模型规模上验证了所提出的 FP8 混合精度框架，训练了大约 1 万亿个 token（有关详细信息，请参阅附录 B.1）。值得注意的是，与 BF16 基线相比，我们的 FP8 训练模型的相对损失误差始终保持在 0.25% 以下，这一水平完全在训练随机性的可接受范围内。

### 3.3.1. Mixed Precision Framework
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28c0ff498cab4a409d85f6d4a3bdf731.png)
### 3.3.2. Improved Precision from Quantization and Multiplication
### 3.3.3. Low-Precision Storage and Communication
## 3.4. Inference and Deployment

我们在 H800 集群上部署了 DeepSeek-V3，其中每个节点内的 GPU 使用 NVLink 互连，集群中的所有 GPU 通过 IB 完全互连。为了同时确保在线服务的服务级别目标 (SLO) 和高吞吐量，我们采用了以下部署策略，将预填充阶段和解码阶段分开。

### 3.4.1. Prefilling
### 3.4.2. Decoding
# 4.Pre-Training
## 4.1 Data Construction

与 DeepSeek-V2 相比，我们通过提高数学和编程样本的比例来优化预训练语料库，同时将多语言覆盖范围扩大到英语和中文之外。此外，我们的数据处理流程经过改进，以最大限度地减少冗余，同时保持语料库的多样性。受 Ding et al. (2024) 的启发，我们实现了文档packing 方法以确保数据完整性，但在训练期间不加入跨样本注意力 mask。最后，DeepSeek-V3 的训练语料由我们的 tokenizer 中的 14.8T 高质量和多样化的 token 组成。

在 DeepSeekCoder-V2 的训练过程中，我们观察到 **Fill-in-Middle (FIM)** 策略不会损害下一个 token 的预测能力，同时使模型能够根据上下文线索准确预测中间文本。与 DeepSeekCoder-V2 保持一致，我们在 DeepSeek-V3 的预训练中也采用了 FIM 策略。具体来说，我们采用 Prefix-Suffix-Middle (PSM) 框架来构造数据，如下所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a091a6c41c924bdc8f5413e62493b174.png)

此结构在文档级别应用，作为预 packing 过程的一部分。FIM 策略以 0.1 的概率被应用，与 PSM 框架一致。

DeepSeek-V3 的分词器采用字节级 BPE，词汇量扩展为 128K 。我们对预分词器和分词器的训练数据进行了修改，以优化多语言压缩效率。此外，与 DeepSeek-V2 相比，新的预分词器引入了组合标点符号和换行符的分词器。然而，当模型处理没有终端换行符的多行提示时，这种技巧可能会引入分词边界偏差，尤其是对于少样本评估提示。为了解决这个问题，我们在训练期间随机拆分了一定比例的此类组合分词器，这使模型能够接触更广泛的特殊情况并减轻这种偏差。

## 4.2 Hyper-Parameters

**Model Hyper-Parameters**。我们将 Transformer 层数设置为 61，隐藏层维度设置为 7168。所有可学习参数均以标准差 0.006 随机初始化。在 MLA 中，我们将注意力头 $n_h$ 的数量设置为 128，每个头的维度 $d_h$ 设置为 128。KV 压缩维度 $d_c$ 设置为 512，query 压缩维度 $d'_c$ 设置为 1536。对于解耦的 query 和 key，我们将每个头的维度 $d^R_h$ 设置为 64。我们用 MoE 层替换除前三层之外的所有 FFN。每个 MoE 层由 1 个共享专家和 256 个路由专家组成，其中每个专家的中间隐藏维度为 2048。在路由专家中，每个 token 将激活 8 个专家，并确保每个 token 最多发送到 4 个节点。多 token 预测深度 $D$ 设置为 1，即除了精确的下一个 token 之外，每个 token 还会预测一个额外的 token。与 DeepSeek-V2 一样，DeepSeek-V3 也在压缩的潜在向量之后使用了额外的 RMSNorm 层，并在宽度瓶颈处乘以额外的缩放因子。在这种配置下，DeepSeek-V3 包含总共 671B 个参数，其中每个 token 激活 37B 个。

**Training Hyper-Parameters**。我们使用 AdamW 优化器，超参数设置为 $\beta_1 = 0.9, \beta_2 = 0.95$ 和 weight_decay = 0.1。我们在预训练期间将最大序列长度设置为 4K，并在​​ 14.8T token上对 DeepSeek-V3 进行预训练。至于学习率调度，我们首先在前 2K 步中将其从 0 线性增加到 $2.2 × 10^{−4}$。然后，我们保持 $2.2 × 10^{−4}$ 的恒定学习率，直到模型消耗 10T 训练token。随后，我们按照余弦衰减曲线逐渐将学习率衰减到 4.3T token中的 $2.2 × 10^{−5}$。在训练最后的 500B 个 token 时，我们在前 333B 个 token 中保持 $2.2 × 10^{−5}$ 的恒定学习率，在剩余的 167B 个 token 中切换到另一个 $7.3 × 10^{−6}$ 的恒定学习率。梯度裁剪范数设置为 1.0。我们采用批量大小调度策略，在训练前 469B 个 token 时，批量大小从 3072 逐渐增加到 15360，然后在剩余的训练中保持 15360。我们利用流水线并行性将模型的不同层部署在不同的 GPU 上，对于每一层，路由的专家将均匀部署在属于 8 个节点的 64 个 GPU 上。对于节点限制路由，每个 token 将发送到最多 4 个节点（即 $M = 4$）。对于无辅助损失的负载平衡，我们将偏差更新速度 $\gamma$ 设置为前 14.3T 个 token 的 0.001，将其余 500B 个 token 的偏差更新速度 $\gamma$ 设置为 0.0。对于平衡损失，我们将 $\alpha$ 设置为 0.0001，以避免任何单个序列中的极端不平衡。MTP 损失权重 $\lambda$ 设置为前 10T 个 token 的 0.3，将其余 4.8T 个 token 的偏差更新速度 $\lambda$ 设置为 0.1。

## 4.3 Long Context Extension
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/86af49d39d5c4cdfa349d32aa861867f.png)

我们采用与 DeepSeek-V2 类似的方法，在 DeepSeek-V3 中启用长上下文功能。在预训练阶段之后，我们应用 YaRN 进行上下文扩展，并执行两个额外的训练阶段，每个阶段包含 1000 个步骤，以逐步将上下文窗口从 4K 扩展到 32K，然后再扩展到 128K。YaRN 配置与 DeepSeek-V2 中使用的配置一致，专门应用于解耦共享key $\textbf k^R_t$。两个阶段的超参数保持不变，即尺度 $s = 40, \alpha = 1、\beta = 32$，比例因子 $\sqrt{t} = 0.1 ln s + 1$。在第一阶段，序列长度设置为 32K，批量大小为 1920。在第二阶段，序列长度增加到 128K，批量大小减小到 480。两个阶段的学习率都设置为 $7.3 × 10^{−6}$，与预训练阶段的最终学习率相匹配。

通过这种两阶段扩展训练，DeepSeek-V3 能够处理长达 128K 的输入，同时保持强劲的性能。图 8 表明，经过监督微调后，DeepSeek-V3 在“大海捞针”(NIAH) 测试中取得了显著的性能，在长达 128K 的上下文窗口长度中表现出一致的稳健性。

## 4.4. Evaluations
### 4.4.1. Evaluation Benchmarks
### 4.4.2. Evaluation Results
## 4.5. Discussion
### 4.5.1. Ablation Studies for Multi-Token Prediction
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55a13aad2d4d492a9252d3a379f64e5b.png)
### 4.5.2 Ablation Studies for the Auxiliary-Loss-Free Balancing Strategy
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea283f30223c4f9990a4ce53af643445.png)
### 4.5.3 Batch-Wise Load Balance VS. Sequence-Wise Load Balance
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/569688c8bd114d8e8925f83a044ac610.png)
# 5. Post-Training
## 5.1 Supervised Fine-Tuning

我们精心整理了我们的指令微调数据集，以包含跨多个领域的 150 万个实例，每个领域都采用根据其特定要求定制的独特数据创建方法。

**Reasoning Data**。对于数学、代码竞赛题、逻辑谜题等推理类数据集，我们利用内部的 DeepSeek-R1 模型生成数据。具体来说，R1 生成的数据虽然准确率高，但也存在过度思考、格式差、篇幅过长等问题。我们的目标是在 R1 生成的推理数据的高准确率与规则格式的推理数据的清晰简洁之间取得平衡。

为了建立我们的方法，我们首先使用有监督微调 (SFT) 和强化学习 (RL) 训练流程，开发针对特定领域（例如代码、数学或一般推理）的专家模型。此专家模型用作最终模型的数据生成器。训练过程涉及为每个实例生成两种不同类型的 SFT 样本：第一种将问题与其原始响应结合在一起，格式为 $<problem, original~response>$，而第二种将系统提示与问题和 R1 响应结合在一起，格式为 $<system~prompt, problem, R1~response>$。

系统提示经过精心设计，包含指导模型生成富含反思和验证机制的响应的指令。在 RL 阶段，即使在没有明确系统提示的情况下，该模型也会利用 high-temperature 采样来生成集成 R1 生成数据和原始数据模式的响应。经过数百个 RL 步骤后，中级 RL 模型学会整合 R1 模式，从而从战略上提高整体性能。

完成强化学习训练阶段后，我们会实施拒绝抽样，为最终模型收集高质量的 SFT 数据，其中专家模型用作数据生成源。此方法可确保最终训练数据保留 DeepSeek-R1 的优势，同时产生简洁有效的响应。

**Non-Reasoning Data**。对于非推理数据，例如创意写作、角色扮演和简单问答，我们利用 DeepSeek-V2.5 来生成响应并聘请人工标注者来验证数据的准确性和正确性。

**SFT Settings**。我们使用 SFT 数据集对 DeepSeek-V3-Base 进行了两个 epoch 的微调，采用余弦衰减学习率调度，从 $5 × 10^{−6}$ 开始逐渐减小到 $1 × 10^{−6}$。在训练期间，每个单个序列都由多个样本 packing 而成。但是，我们采用样本屏蔽策略来确保这些示例保持孤立且相互不可见。

## 5.2 Reinforcement Learning
### 5.2.1 Reward Model
### 5.2.2 Group Relative Policy Optimization
