# Muon is Scalable for LLM Training

论文链接：https://arxiv.org/pdf/2502.16982

代码链接：

## 摘要

最近，基于矩阵正交化的 Muon 优化器在训练小规模语言模型方面展现出了强大的实力，但其对大规模模型的可扩展性尚未得到验证。**我们提出了两种扩展 Muon 的关键技术**：(1) 添加权重衰减；(2) 精确调整每个参数的更新尺度。这些技术使得 Muon 无需超参数调优即可直接用于大规模训练。扩展性实验表明，与采用计算最优训练的 AdamW 相比，Muon 的计算效率提高了约 2 倍。基于这些改进，我们推出了 **Moonlight**，一个使用 Muon 训练的 3B/16B 参数混合专家 (MoE) 模型，该模型使用 5.7T 个 token 进行训练。我们的模型改进了当前的帕累托前沿，与之前的模型相比，在训练浮点运算次数大幅减少的情况下实现了更优的性能。我们开源了内存优化且通信高效的分布式 Muon 实现。此外，我们还发布了预训练、指令微调和中间检查点，以支持未来的研究。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/42172b79b7fa4b79bbdd725c67f00bc2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

大型语言模型（LLM）的快速发展（OpenAI et al. 2024; DeepSeek-AI et al. 2024; Grattafiori et al. 2024; Gemini Team et al. 2024）显著推动了通用人工智能的进步。然而，由于缩放定律，训练功能强大的 LLM 仍然是一个计算密集型且资源消耗巨大的过程。优化器在高效训练 LLM 方面发挥着至关重要的作用，其中 Adam 及其变体 AdamW 是大多数大规模训练的标准选择。

近年来，优化算法的发展展现出超越 AdamW 算法的训练效率提升潜力。其中，K. Jordan 等人于 2024 年提出的 Muon 算法，利用牛顿-舒尔茨迭代法，通过正交化的梯度动量来更新矩阵参数。Muon 的初步实验在小规模语言模型训练中取得了令人瞩目的成果。**然而，正如本博客所讨论的，仍存在一些关键挑战尚未解决**：（1）如何有效地将基于矩阵正交化的优化器扩展到拥有数十亿参数、训练数据量达数万亿的大型模型；（2）如何在分布式环境下计算近似正交化；（3）此类优化器能否在包括预训练和有监督微调（SFT）在内的不同训练阶段保持泛化能力。

在本技术报告中，我们针对这些挑战提出了一项全面的研究。我们的工作以 Muon 为基础，并系统地识别和解决了其在大规模训练场景中的局限性。我们的技术贡献包括：
- **Analysis for Effective Scaling of Muon**。通过深入分析，我们发现权重衰减在 Muon 的可扩展性中起着至关重要的作用。此外，我们提出了对 Muon 参数更新规则的尺度调整。这些调整使得 Muon 无需超参数调优即可开箱即用，并显著提高了训练稳定性。
- **Efficient Distributed Implementation**。我们开发了 Muon 的分布式版本，并采用了 ZeRO-1 式的优化，在保持算法数学特性的同时，实现了最佳的内存效率和降低的通信开销。
- **Scaling Law Validation**。我们进行了缩放定律研究，将 Muon 与强大的 AdamW 基线进行了比较，结果表明 Muon 具有更优越的性能 (1​​a)。基于缩放定律结果，Muon 达到了可比的水平。

我们全面的实验表明，Muon 可以有效地取代 AdamW，成为大规模 LLM 训练的实际标准优化器，并在训练效率和模型性能方面均有显著提升。基于这项工作，我们发布了 Moonlight，这是一个使用 Muon 训练的 16B 参数 MoE 模型，同时还提供了我们的实现和中间训练检查点，以促进对 LLM 可扩展优化技术的进一步研究。

## 2.Methods

### 2.1 Background

**The Muon Optimizer**。最近，一种名为 Muon 的优化器被提出，用于优化以矩阵形式表示的神经网络权重。在第 $t$ 次迭代中，给定当前权重 $\textbf W_{t−1}$、动量 $µ$、学习率 $η_t$ 和目标函数 $\mathcal L_t$，Muon 优化器的更新规则可以表述如下：

```math
\begin{array}{cc}
\textbf M_t=µ\textbf M_{t-1}+\nabla\mathcal L_t(\textbf W_{t-1})\\
\textbf O_t=\text{Newton-Schulz}(\textbf M_t)\\
\textbf W_t=\textbf W_{t-1} - η_t\textbf O_t
\end{array}\tag{1}
```

这里，$\textbf M_t$ 是第 $t$ 次迭代的梯度动量，当 $t = 0$ 时设为零矩阵。在公式 1 中，采用牛顿-舒尔茨迭代法来近似求解 $(\textbf M_t\textbf M^T_t)^{−1/2}\textbf M_t$。令 $\textbf U\textbf Σ\textbf V^T = \textbf M_t$ 为 $\textbf M_t$ 的奇异值分解 (SVD)，则有 $(\textbf M_t\textbf M^T_t)^{−1/2}\textbf M_t = \textbf U\textbf V^T$，这使得 $\textbf M_t$ 正交化。直观地说，**正交化可以确保更新矩阵同构，防止权重沿少数几个主要方向学习**。

**Newton-Schulz Iterations for Matrix Orthogonalization**。公式 1 的计算采用迭代方法。初始时，我们设定 $\textbf X_0 = \textbf M_t/∥\textbf M_t∥_F$。然后，在每次迭代 $k$ 中，我们按如下方式将 $\textbf X_{k-1}$ 更新到 $\textbf X_{k}$：

```math
\textbf X_k=a\textbf X_{k-1}+b(\textbf X_{k-1}\textbf X^T_{k-1})\textbf X_{k-1}+c(\textbf X_{k-1}\textbf X^T_{k-1})^2\textbf X_{k-1}\tag{2}
```

其中 $\textbf X_N$ 是经过 $N$ 次迭代后的结果。这里 $a, b, c$ 是系数。为了确保方程 2 的正确收敛，我们需要调整系数，使多项式 $f(x) = ax + bx^3 + cx^5$ 在 1 附近有一个不动点。在 K. Jordan et al. 2024 的原始设计中，系数设置为 $a = 3.4445, b = −4.7750, c = 2.0315$，以使迭代过程在初始奇异值较小时能够更快地收敛。本文沿用了相同的系数设置。

**Steepest Descent Under Norm Constraints**。Bernstein et al. 2024 提出将深度学习中的优化过程视为*范数约束下的最速下降法*。从这个角度来看，Muon 和 Adam 的区别在于范数约束的不同。Adam 是在动态调整的范数约束（基于 Max-of-Max 范数）下进行的最速下降，而 Muon 则提供了一个静态的 Schatten-p 范数约束（p 为某个较大的值）。当公式 1 被精确计算时，Muon 提供的范数约束将是谱范数。神经网络的权重被用作输入空间或隐藏空间上的算子，而输入空间或隐藏空间通常是（局部）欧几里得空间，因此权重的范数约束应该是诱导算子范数（或权重矩阵的谱范数）。从这个意义上讲，Muon 提供的范数约束比 Adam 提供的更合理。

### 2.2 Scaling Up Muon

<img
  src="https://i-blog.csdnimg.cn/direct/7b7b0ca3b33c4cd4bddfbf473635b2a2.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Weight Decay**。尽管如 K. Jordan et al. 2024 所述，Muon 在小规模模型上的表现显著优于 AdamW，但我们发现，当模型规模扩大到包含更多 token 时，性能优势会逐渐减弱。我们观察到，权重和层输出的 RMS 值都会持续增长，超出 bf16 的高精度范围，这可能会损害模型的性能。为了解决这个问题，我们将标准的 AdamW 权重衰减机制引入到 Muon 中。

```math
\textbf W_t=\textbf W_{t-1}-η_t(\textbf O_t+\lambda\textbf W_{t-1})\tag{3}
```

我们对启用和禁用权重衰减的 Muon 进行了实验，以了解权重衰减对 LLM 训练动态的影响。基于我们在第 3.2 节中对缩放律的研究，我们使用 100B 个 token（约为最优训练 token 的 5 倍）训练了一个 800M 参数的模型。图 2 展示了使用 AdamW、未启用权重衰减的 Muon 以及启用权重衰减的 Muon 训练的模型的验证损失曲线。虽然未启用权重衰减的 Muon 初始收敛速度更快，但我们观察到一些模型权重会随着时间的推移而增长过大，这可能会限制模型的长期性能。启用权重衰减解决了这个问题——结果表明，启用权重衰减的 Muon 的性能优于未启用权重衰减的 Muon 和 AdamW，**在过拟合状态下获得了更低的验证损失**。因此，我们将更新规则调整为公式 3，其中 $λ$ 为权重衰减率。

**Consistent update RMS**。Adam 和 AdamW 的一个重要特性是它们保持理论更新均方根误差在 1 左右。然而，我们证明，根据以下引理，Muon 的更新均方根误差会随参数形状而变化：

*Lemma 1*。对于形状为 $[A, B]$ 的满秩矩阵参数，其理论 Muon 更新 RMS 为 $\sqrt{1/ max(A, B)}$。

证明见附录 A。我们在训练过程中监测了 Muon 的更新均方根误差，发现它通常接近上述理论值。我们注意到，当模型规模扩大时，这种不一致性可能会带来问题：
- 当 $max(A, B)$ 过大时，例如对于稠密的 MLP 矩阵，更新会变得太小，从而限制模型的表示能力，导致性能欠佳；
- 当 $max(A, B)$ 太小时，例如将 GQA 或 MLA 中的每个 KV 头视为单独的参数时，更新会变得太大，从而导致训练不稳定，并导致性能不佳。

为了保持不同形状矩阵之间更新 RMS 的一致性，我们建议将每个矩阵的 Muon 更新按其 $\sqrt{max(A, B)}$ 进行缩放，以抵消 Lemma 1 的影响。第 3.1 节中的实验表明，该策略有利于优化。

**Matching update RMS of AdamW**。Muon 旨在更新基于矩阵的参数。在实践中，AdamW 通常与 Muon 结合使用，以处理非矩阵参数，例如 RMSNorm、LM Head 和 Embedding 参数。我们希望优化器的超参数（学习率 $η$ 和权重衰减 $λ$）能够在矩阵参数和非矩阵参数之间共享。

我们建议将 Muon 的更新 RMS 调整至与 AdamW 相似。根据经验观察，AdamW 的更新 RMS 通常在 0.2 到 0.4 之间。因此，我们通过以下调整将 Muon 的更新 RMS 缩放到该范围内：

```math
\textbf W_t=\textbf W_{t-1}- η_t(0.2\cdot\textbf O_t\cdot\sqrt{max(A,B)}+\lambda\textbf W_{t-1})\tag{4}
```

我们通过实验结果验证了这一选择（详见附录 A）。此外，我们强调，通过这种调整，Muon 可以直接复用为 AdamW 优化的学习率和权重衰减机制。

**Other Hyper-parameters**。Muon 还包含另外两个可调超参数：牛顿-舒尔茨迭代步数和动量 $µ$。我们通过实验观察到，当 $N=10$ 时，迭代过程会比 $N=5$ 时产生更精确的正交化结果，但性能并不会因此得到提升。因此，为了提高效率，本文将 $N$ 设置为 5。我们没有观察到调整动量能带来持续的性能提升，因此我们选择了 $0.95$，与 K. Jordan 等人 2024 年的研究结果相同。

### 2.3 Distributed Muon

<img
  src="https://i-blog.csdnimg.cn/direct/6d3889de684c41fda7f2b265f17934b0.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**ZeRO-1 and Megatron-LM**。Rajbhandari et al. 2020 提出了 ZeRO-1 技术，该技术将计算量巨大的优化器状态（例如主权重、动量）分布在整个集群中。Megatron-LM 将 ZeRO-1 集成到其原生并行设计中。基于 Megatron-LM 先进的并行策略，例如张量并行 (TP)、流水线并行 (PP)、专家并行 (EP) 和数据并行 (DP)，ZeRO-1 的通信负载可以从遍历整个分布式集群减少到仅通过数据并行组进行通信。

**Method**。ZeRO-1 对 AdamW 来说效率很高，因为它逐元素地计算更新。然而，Muon 需要完整的梯度矩阵才能计算更新。因此，原始的 ZeRO-1 不能直接应用于 Muon。

我们提出了一种基于 ZeRO-1 的新型分布式 Muon 解决方案，称为分布式 Muon。分布式 Muon 沿用 ZeRO-1 对 DP 上的优化器状态进行划分，并且与标准的 Zero-1 AdamW 优化器相比，引入了两个额外的操作：

1. *DP Gather*。对于局部 DP 分区主权重（$1/DP$ 是模型权重的大小），此操作是将相应的分区梯度收集到一个完整的梯度矩阵中。
2. *Calculate Full Update*。完成上述收集后，按照第 2.1 节所述，对完整的梯度矩阵执行牛顿-舒尔茨迭代步骤。请注意，我们将丢弃完整更新矩阵的一部分，因为我们只需要与局部参数对应的分区来执行更新。

分布式 Muon 的实现如算法 1 所示。分布式 Muon 引入的附加操作以蓝色标出。

**Analysis**。我们从几个方面将分布式 Muon 与经典的基于 ZeRO-1 的分布式 AdamW 算法（为简便起见，简称分布式 AdamW）进行了比较：
- *Memory Usage*。Muon 仅使用一个动量缓冲区，而 AdamW 使用两个动量缓冲区。因此，Muon 优化器使用的额外内存是分布式 AdamW 的一半。
- *Communication Overhead*。对于每个设备，额外的 DP 收集仅由本地 DP 分区参数 $\textbf p$ 决定。因此，其通信成本低于 $\textbf G$ 的 reduce-scatter 或 $\textbf P$ 的 all-gather。此外，Muon 仅需 bf16 中的 Newton-Schulz 迭代步骤，从而进一步将通信开销降低至 fp32 的 50%。总体而言，分布式 Muon 的通信工作负载是分布式 AdamW 的 (1, 1.25]。其上限计算为：分布式 Muon 的通信成本为 4 (fp32 $\textbf G$ reduce-scatter) + 2 (bf16 Muon gather) + 4 (fp32 $\textbf P$ all-gather)，而分布式 AdamW 的通信成本为 4 + 4。在实践中，由于我们通常使用多个 DP 进行训练，因此经验上的额外成本通常更接近下限 1.5。
- *Latency*。分布式 Muon 算法的端到端延迟比分布式 AdamW 算法更大，因为它引入了额外的通信，并且需要运行牛顿-舒尔茨迭代步骤。然而，这并不是一个严重的问题，因为 (a) 获得良好结果只需要大约 5 次牛顿-舒尔茨迭代步骤（详见 2.2 节），(b) 优化器造成的端到端延迟与模型的前向-后向传播时间相比可以忽略不计（例如，通常为 1% 到 3%）。此外，一些工程技术，例如重叠采集和计算，以及重叠优化器减散和参数采集，可以进一步降低延迟。

在我们的分布式集群中训练大规模模型时，分布式 Muon 相较于 AdamW 算法几乎没有明显的延迟开销。我们即将发布一个 pull request，将分布式 Muon 集成到开源项目 Megatron-LM 中。

## 3.Experiments

### 3.1 Consistent Update RMS

如第 2.2 节所述，我们的目标是使所有矩阵参数的更新均方根 (RMS) 保持一致，并与 AdamW 的 RMS 保持一致。我们尝试了两种方法来控制不同参数间的 Muon 更新 RMS，并将它们与仅保持与 AdamW 一致的 RMS 的基线方法进行了比较：

1. **Baseline**。为了与 AdamW 保持一致的更新均方根误差，我们将更新矩阵乘以 $0.2 ·\sqrt{H}$（$H$ 为模型隐藏层大小）。需要注意的是，对于大多数矩阵，$max(A, B)$ 等于 H。
```math
\textbf W_t=\textbf W_{t-1}-η_t(0.2\cdot\textbf O_t\cdot\sqrt{H}+\lambda\textbf W_{t-1})\tag{5}
```
2. **Update Norm**。我们可以直接对通过牛顿-舒尔茨迭代计算出的更新进行归一化，使其均方根严格变为 0.2；
```math
\textbf W_t=\textbf W_{t-1}-η_t(0.2\cdot\textbf O_t/\text{RMS}(\textbf O_t)+\lambda\textbf W_{t-1})\tag{6}
```
3. **Adjusted LR**。对于每个更新矩阵，我们可以根据其形状，将其学习率按 $0.2 · \sqrt{max(A, B)}$ 的因子进行缩放。
```math
\textbf W_t=\textbf W_{t-1}-η_t(0.2\cdot\textbf O_t\cdot\sqrt{max(A,B)}+\lambda\textbf W_{t-1})\tag{7}
```

**Analysis**。我们设计了实验来展示 Muon 更新 RMS 在早期训练阶段的影响，因为我们观察到，在大规模模型训练中，意外行为会很快出现。我们使用 3.2 节中描述的 800M 小规模模型进行了实验。当矩阵维度差异增大时，更新 RMS 不一致的问题会更加突出。为了突出这个问题以便进一步研究，我们对模型架构进行了微调，将 Swiglu MLP 替换为标准的 2 层 MLP，并将其矩阵参数的形状从 $[H, 2.6H]$ 改为 $[H, 4H]$。我们评估了模型的损失，并监测了几个参数的 RMS。我们在训练了 4B token 后评估了模型，训练计划为 20B token。从表1中，我们观察到了一些有趣的发现：
1. *Update Norm* 和 *Adjusted LR* 均比 *Baseline* 表现更好；
2. 对于形状为 $[H, 4H]$ 的 MLP 权重矩阵，*Update Norm* 和 *Adjusted LR* 的权重 RMS 均比基线值大约翻了一倍。这是合理的，因为 $\sqrt{max(H, 4H)}/\sqrt{H} = 2$，所以 *Update Norm* 和 *Adjusted LR* 的更新 RMS 大约是基线值的两倍；
3. 对于形状为 $[H, H]$ 的注意力查询权重矩阵，*Update Norm* 仍然对更新进行归一化，而 *Adjusted LR* 则不进行归一化，因为 $\sqrt{max(H, H)}/\sqrt{H} = 1$。因此，*Adjusted LR* 的权重 RMS 与 Baseline 相似，但 *Update Norm* 的权重 RMS 较大，与 MLP 类似。 

基于这些发现，我们选择 *Adjusted LR* 方法用于未来的实验，因为它成本较低。

### 3.2 Scaling Law of Muon

<img
  src="https://i-blog.csdnimg.cn/direct/2e247ae91ab24f2e91a14c9e60cdf7aa.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

为了与 AdamW 进行公平的比较，我们对一系列 Llama 架构的密集模型进行了缩放律实验。**构建一个强大的基线对于优化器研究至关重要**。因此，我们按照计算最优训练设置，对 AdamW 的超参数进行了网格搜索（网格搜索实验详见附录 B）。模型架构和超参数的详细信息见表 2。对于 Muon，如第 2.2 节所述，由于我们已将 Muon 的更新 RMS 与 AdamW 相匹配，因此我们直接复用了 AdamW 基线的最优超参数。

拟合的缩放律曲线如图 3 所示，拟合方程详见表 3。如图 1a 所示，在计算最优设置下，Muon 仅需约 52% 的训练 FLOPs 即可达到 AdamW 的性能。

### 3.3 Pretraining with Muon

### 3.4 Dynamics of Singular Spectrum

<img
  src="https://i-blog.csdnimg.cn/direct/ebff3b89abd843e89d970514fb086766.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

为了验证 Muon 能够以更多样化的方向优化权重矩阵这一直觉，我们对使用 Muon 和 AdamW 训练得到的权重矩阵进行了谱分析。对于一个奇异值为 $\sigma=(\sigma_1,\sigma_2,\cdots,\sigma_n)$ 的权重矩阵，我们按照如下方式计算该矩阵的 SVD 熵：

```math
H(\sigma)
=-\frac{1}{\log n}
\sum_{i=1}^{n}
\frac{\sigma_i^2}{\sum_{j=1}^{n}\sigma_j^2}
\log
\frac{\sigma_i^2}{\sum_{j=1}^{n}\sigma_j^2}
```

如图 4 所示，我们可视化了在使用 1.2T tokens 进行预训练过程中，不同训练 checkpoint 下各权重矩阵的平均 SVD 熵。可以看到，在所有训练 checkpoint 以及所有权重矩阵分组中，Muon 的 SVD 熵都高于 AdamW，这验证了 Muon 能够为权重矩阵提供更加多样化更新谱的直觉。这种差异在用于专家选择的 router 权重上更加显著，这表明 mixture-of-expert 模型能够从 Muon 中获得更大的收益。

此外，我们还可视化了在使用 1.2T tokens 训练得到的 checkpoint 中，每个权重矩阵的奇异值分布，如附录 F 所示。我们发现，对于超过 90% 的权重矩阵，使用 Muon 优化时的 SVD 熵高于 AdamW，这为 Muon 在探索多样化优化方向方面具有更强能力提供了有力的经验证据。

### 3.5  Supervised Finetuning (SFT) with Muon

#### 3.5.1 Ablation Studies on the Interchangeability of Pretrain and SFT Optimizers

#### 3.5.2 SFT with Muon on public pretrained models

## 4.Discussions

## 5.Conclusions
