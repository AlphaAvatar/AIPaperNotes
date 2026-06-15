# Muon: An optimizer for hidden layers in neural networks

论文链接：https://kellerjordan.github.io/posts/muon/

代码链接：https://github.com/KellerJordan/Muon

## 摘要

Muon 是一种用于神经网络隐藏层的优化器。它被用于 NanoGPT 和 CIFAR-10 速通的当前训练速度记录中。

许多使用 Muon 的实证结果已经发表，因此本文将主要关注 Muon 的设计。首先，我们将定义 Muon，并概述其迄今为止取得的实证结果。然后，我们将详细讨论其设计，包括与先前研究的联系以及我们对它有效机制的最佳理解。最后，我们将讨论优化研究中的证据标准。

## Definition

**Muon 是一个用于优化神经网络隐藏层 2D 参数的优化器**。它的定义如下：

<img
  src="https://i-blog.csdnimg.cn/direct/376683ef7ba74b75a25141a5b46e1f3c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

其中，`NewtonSchulz5` 定义为以下牛顿-舒尔茨矩阵迭代（Bernstein & Newhouse, 2024; Higham, 2008; Björck and Bowie, 1971; Kovarik, 1970）：

```python
# Pytorch code
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

这里（https://github.com/KellerJordan/Muon）提供了一个可直接使用的 Muon PyTorch 实现。这里（https://github.com/KellerJordan/modded-nanogpt/blob/973030408364f8738b4ad9e8f912d8cbbf56e4d4/train_gpt2.py#L455）提供了一个在当前 NanoGPT 速通记录中的示例用法。

**使用 Muon 训练神经网络时，网络的标量和向量参数以及输入层和输出层都应使用标准方法（例如 AdamW）进行优化**。Muon 可以通过展平卷积参数的最后三个维度（https://github.com/KellerJordan/cifar10-airbench/blob/0e6f9614572d7e8e3c259905aebc7196f91d5d79/research/clean_muon.py#L95）来处理 4D 卷积参数。

## Results

Muon 取得了以下实证成果：
- 在 CIFAR-10 数据集上，训练准确率达到 94% 的速度记录从 A100 3.3 秒提升至 2.6 秒。
- 在 FineWeb 数据集（一项被称为 NanoGPT 速通的竞争性任务）上，训练损失达到 3.28 的速度记录提升了 1.35 倍。
- 在参数规模扩展至 774M 和 1.5B 的情况下，训练速度持续提升。
- 在 HellaSwag 数据集上，仅用 8xH100 10 个小时就将一个 1.5B 参数的 Transformer 模型训练到了 GPT-2 XL 的性能水平。而使用 AdamW 达到同样的效果则需要 13.3 小时。

以下是针对 NanoGPT 速通的不同强力优化器的比较：

<img
  src="https://i-blog.csdnimg.cn/direct/bcb7eab9f9dd4df9a7a830e4808a87ad.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

<img
  src="https://i-blog.csdnimg.cn/direct/497991edb7d544dcb12cc0d13ceafbf8.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

此外，这里还对 Muon 和 AdamW 进行了比较，以训练一个 1.5B 参数的语言模型。这两个优化器都经过了调优。

<img
  src="https://i-blog.csdnimg.cn/direct/528d88f064764109acd260caebe3b906.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## The design of Muon

本节描述并分析 Muon 的设计。

**Muon**（MomentUm Orthogonalized by Newton-Schulz）通过获取 SGD 动量生成的更新来优化 2D 神经网络参数，然后在将更新应用于参数之前，对每个更新应用牛顿-舒尔茨 (NS) 迭代作为后处理步骤。

**NS 迭代的功能是近似正交化更新矩阵**，即应用以下操作：

```math
Ortho(G)=arg\mathop{min}\limits_{O}\{||O-G||_F: either~O^TO=I~or~OO^T=I\}
```

换句话说，NS 迭代有效地将 SGD 动量更新矩阵替换为与其最接近的**半正交矩阵**。这等价于用 $UV^T$ 替换更新矩阵，其中 $USV^T$ 是其奇异值分解（SVD）。

## Why is it good to orthogonalize the update?

我们首先想指出，一个合理的答案是：这样做没问题？（Shazeer 2020）

<img
  src="https://i-blog.csdnimg.cn/direct/62cae2f5c2d940f884a927091354cce6.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>
但是，对于源自 Bernstein & Newhouse (2024) 对 Shampoo 的分析 (Gupta et al. 2018) 的理论动机，请参阅与 Shampoo 的关系部分（https://kellerjordan.github.io/posts/muon/#shampoo）。

为了提供经验性的论证，我们观察到，基于人工检查，SGD 动量和 Adam 优化器对 Transformer 神经网络中二维参数的更新通常具有非常高的条件数。也就是说，它**们几乎是低秩矩阵，所有神经元的更新都主要由少数几个方向决定**。我们推测，正交化有效地增大了其他“稀有方向”的尺度，这些方向虽然在更新中幅度较小，但对学习至关重要。

## Eliminating alternatives to NS iteration

除了 NS 迭代法之外，还有其他几种方法可以对矩阵进行正交化。本小节将解释为什么我们没有使用其中的两种方法。有关更完整的可用方法列表，请参阅 Bernstein & Newhouse (2024) 的附录A。

**SVD**（即计算更新的 $USV^T$ 分解，然后用 $UV^T$ 替换更新）很容易理解，但我们不使用它，因为它太慢了。

**耦合牛顿迭代法** (Guo and Higham, 2006; Iannazzo, 2006) 在 Shampoo 的实现中 (Gupta et al. 2018; Anil et al. 2020; Shi et al., 2023) 用于计算四次方根逆，并且可以很容易地进行调整以实现正交化。但我们没有使用它，因为我们发现它必须至少以 float32 精度运行才能避免数值不稳定，这使得它在现代 GPU 上运行速度很慢。

相比之下，我们发现牛顿-舒尔茨迭代法（Bernstein & Newhouse, 2024; Higham, 2008; Björck and Bowie, 1971; Kovarik, 1970）可以在 bfloat16 中稳定运行。因此，我们选择它们作为正交化更新的首选方法。

## Proving that NS iteration orthogonalizes the update

为了理解为什么 NS 迭代能够正交化更新，令 $G=USV^T$ 为 SGD 动量法生成的更新矩阵的奇异值分解（SVD）。然后，使用系数 $(a,b,c)$ 运行一次 NS 迭代，得到以下输出：

```math
\begin{aligned}
G'
&:=aG+b(GG^T)G+c(GG^T)^2G\\
&=(aI+b(GG^T)+c(GG^T)^2)G\\
&=(aI+bUS^2U^T+cUS^4U^T)USV^T\\
&=U(aS+bS^3+cS^5)V^T
\end{aligned}
```

一般来说，如果我们定义五次多项式 $\varphi(x)=ax+bx^3+cx^5$，那么应用 $N$ 步 NS 迭代，系数为 $(a,b,c)$，即可得到输出 $U\varphi^N(S)V^T$，其中 $\varphi^N(S)$ 表示对构成 $S$ 对角线的奇异值逐元素 $\varphi$ 应用 $N$ 次。

因此，为了保证 NS 迭代收敛到 $Ortho(G)=UV^T$，我们只需要 (1) 确保 $S$ 的初始值在 $[0,1]$ 范围内，以及 (2) 选择系数，使得 $N\to \infty$ 对所有 $x\in[0,1]$ 都等于 $\varphi^N(x)\to 1$。

为了满足第一个条件，我们只需在开始 NS 迭代之前将 $G$ 替换为 $G/||G||_F$。这种重新缩放是良性的，因为 $Ortho(cG)=Ortho(G)$。

当 $N\to\infty$ 时为了满足 $\varphi^N(x)\to 1$ 的要求，我们有一定的自由度，因为有很多 $(a,b,c)$ 方案可以满足这个条件。稍后我们会对这个方案进行优化，但现在我们通过下图展示，简单的基线 $(a,b,c)=(2,-1.5,0.5)$ 方案已经能够满足要求。

<img
  src="https://i-blog.csdnimg.cn/direct/65959f7064144c438b961f2690733df1.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## Tuning the coefficients

虽然 NS 系数 $(a,b,c)=(2,-1.5,0.5)$ 在正交化更新方面效果很好，但还可以进一步调整它们，**以减少我们需要运行的 NS 迭代步骤的数量**。

为了调整系数 $(a,b,c)$，我们需要考虑以下几点：
1. 我们希望使 $a$ 尽可能大，因为 $\varphi '(0)=a$ 表明该系数控制着初始奇异值较小时的收敛速度。
2. 对于每个 $x\in[0,1]$，我们希望 $\varphi^N(x)$ 在 $N\to\infty$ 时收敛到 $[1-\epsilon,1+\epsilon]$ 范围内的值，以便 NS 迭代的结果不会远离 $Otrho(G)$。

令人惊讶的是，经验表明，对于基于 Muon 的训练，$\epsilon$ 值可以高达 0.3 左右而不会损害损失曲线。因此，我们的目标是在 $lim_{N\to\infty}\varphi^N(x)\in [0.7,1.3]$ 约束下最大化 $a$。

解决这个约束优化问题的方法有很多种。我们采用了一种基于梯度的特定方法，最终得到了系数 $(3.4445,-4.7750,2.0315)$，这些系数用于 Muon 的最终设计。这些系数的变化情况如下图所示。请注意 $x=0$ 附近陡峭的增长。

<img
  src="https://i-blog.csdnimg.cn/direct/c10c67c01ad24c5babad5a9b2db21570.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

在我们的实验中，当使用这些系数的 Muon 来训练 Transformer 语言模型和小卷积网络时，只需运行 NS 迭代 5 步即可。

我们也考虑过使用三阶和七阶多项式进行 NS 迭代，但发现这些多项式无法进一步改善运行时间开销。

## Runtime analysis

本节我们将分析 Muon 的运行时间和内存需求。

**在应用 NS 迭代之前，Muon 只是标准的 SGD 动量，因此它具有相同的内存需求**。

对于网络中的每个 $n\times m$ 矩阵参数（不妨设 $m\le n$），NS迭代的每一步都需要 $2(2nm^2+m^3)$ 矩阵乘法 FLOPs，对于方形参数矩阵而言，这至多为 $6nm^2$。因此，与 SGD 相比，Muon 所需的额外 FLOPs 至多为 $6Tnm^2$，其中 $T$ 是 NS 迭代次数（通常我们使用 $T=5$）。

如果该参数被用于参数化线性层，则执行训练步骤（即前向传播和后向传播）所用的 FLOP 基准量为 $6nmB$，其中 $B$ 是该步骤中通过该层的输入数量。

因此，Muon 的 FLOP 开销占比至多为 $Tm/B$，其中 $m$ 为模型维度，$B$ 为一个批次内 token 数大小，$T$ 为 NS 迭代步数（通常为 $T=5$）。

现在我们针对两个具体的训练场景计算这种开销：NanoGPT speedrunning 和 Llama 405B 训练。
1. 对于当前的 NanoGPT speedrunning 记录，模型维度为 $m=768$，每批次的 token 数为 $B=524288$。因此，开销为 $5*768/524288=0.7\%$。
2. 对于 Llama 405B 的训练，模型维度为 $M=16384$，每个批次的 token 数为 $16000000$。因此，使用 Muon 进行此训练的开销为 $5*16384/16000000=0.5\%$。

**我们得出结论，对于典型的 LM 训练场景，无论规模大小，Muon 的 FLOP 开销都低于 1%**。

## Relationship to prior optimizers

### Shampoo

Shampoo 优化器定义如下：

<img
  src="https://i-blog.csdnimg.cn/direct/916ed7c5af0a4b83b465586a75c35d5b.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

如果移除预条件子累积，则 Bernstein & Newhouse (2024) 观察到更新将变为如下形式（另见 Anil (2024a)）：

```math
\begin{aligned}
W_{t+1}
&= W_t - \eta (G_t G_t^\top)^{-1/4} G_t (G_t^\top G_t)^{-1/4} \\
&= W_t - \eta (U S^2 U^\top)^{-1/4} (U S V^\top) (V S^2 V^\top)^{-1/4} \\
&= W_t - \eta (U S^{-1/2} U^\top)(U S V^\top)(V S^{-1/2} V^\top) \\
&= W_t - \eta U S^{-1/2} S S^{-1/2} V^\top \\
&= W_t - \eta U V^\top
\end{aligned}
```

这就是正交化梯度。如果我们在正交化之前加入动量，就可以恢复 Muon 更新，**Shampoo 由于使用了四次方根逆运算而不是牛顿-舒尔茨迭代，因此会增加运行时间和浮点运算开销**。

因此，可以将动量关闭的 Muon 解释为一种“瞬时”或“无积累”的 Shampoo 优化器。

### Orthogonal-SGDM

Tuddenham et al. (2022) 提出了一种优化神经网络的方法：首先通过奇异值分解（SVD）对梯度进行正交化，然后对结果施加动量，最后将动量项作为更新项，并将这种优化器称为 Orthogonal-SGDM。这与 Muon 类似，**区别在于 Muon 将动量施加在正交化之前，我们发现这种方法在实验中表现更佳**；此外，Muon 使用牛顿-舒尔茨迭代法而非 SVD 进行更高效的正交化。在他们性能最佳的实验设置中（表3），Tuddenham et al. (2022) 报告称，他们的方法不如经过良好调优的标准 SGD-Momentum 算法，这或许可以解释为什么这篇论文在本文发表之前没有被引用。

### Stochastic spectral descent and RMSspectral

然而，Carlson et al. (2015a, 2016) 的工作中早有基于正交化的优化实例。他们提出利用奇异值分解（SVD）对受限玻尔兹曼机和离散图模型的梯度估计进行正交化，然后按核范数进行缩放，从而优化这些模型，并将此方法称为 **stochastic spectral descent**。此外，Carlson et al. (2015b) 提出了一种结合随机谱下降法和均方根传播算法（RMSprop）的混合方法，称为 **RMSspectral**，用于优化前馈神经网络。考虑到需要加速正交化过程，RMSspectral 使用随机化 SVD 而非完全 SVD 来近似正交化操作。与 Muon 相比，这些早期基于正交化的优化器使用 SVD 变体而非牛顿-舒尔茨迭代进行正交化，并且缺乏任何形式的动量。**我们发现，使用动量对于获得最佳的实验性能至关重要**。

## Empirical considerations

由于设计原因，Muon 仅适用于二维参数（以及通过展平处理的卷积滤波器），因此网络中其余的标量和向量参数必须使用标准方法（例如 AdamW）进行优化。**经验表明，即使输入和输出参数通常是二维的，使用 AdamW 进行优化也至关重要**。尤其是在训练 Transformer 模型时，为了获得最佳性能，应将 AdamW 用于嵌入层和最终分类器头部。嵌入层的优化动态与其他层不同，这符合模块范数理论。然而，输出层的这种动态也与其他层不同，这似乎并非理论推论，而是由经验驱动的。

另一个纯粹的经验性结果是，在我们测试过的所有情况下，**使用 Nesterov 式动量来计算 Muon 函数的效果都比使用普通的 SGD 动量要好一些**。因此，我们已将其设为公开 Muon 实现中的默认设置。

第三个结果是，如果将 Muon 分别应用于 Transformer 的 Q、K、V 参数，而不是像默认情况下那样将它们一起应用于 Transformer 实现（将 QKV 参数化为单个线性层，其输出被分割），则 Muon 在优化 Transformer 方面效果更好。

## Discussion: Solving the undertuned baseline problem with the competitive task framework

如今，神经网络优化研究文献中充斥着大量号称超越 AdamW 的优化器，它们往往优势巨大，却从未被业界广泛采用。我知道这听起来有些耸人听闻。

**鉴于神经网络训练领域每年投入数十亿美元，而整个行业都在竭力寻找降低成本的方法，我们可以推断，问题出在研究界，而非潜在的采用者身上**。也就是说，研究本身存在问题。仔细审视每篇论文，我们会发现最常见的罪魁祸首是糟糕的基线：论文往往在将 AdamW 基线与新提出的优化器进行比较之前，没有对其进行充分的调优。

我想指出的是，那些声称性能大幅提升却无法复现或达不到预期效果的新方法的发表并非无害，因为这浪费了大量研究人员和小型实验室的时间、金钱和精力，他们每天都在尝试复现和改进这些方法，却屡屡碰壁，最终失望而归。

为了解决这个问题，我建议采用以下证据标准：**研究界应要求，只要有可能，新的神经网络训练方法都应在竞争性训练任务中证明其成功**。

**竞争性任务通过两种方式解决了基线调优不足的问题**。首先，竞争性任务中的基线是先前的记录，如果该任务很热门，那么之前的记录很可能已经调优良好。其次，即使之前的记录调优不佳（这种情况发生的概率很低），也可以通过新的记录进行自我修正，将训练方法恢复到标准方法。之所以可行，是因为标准方法通常有快速的硬件优化实现，而新方法通常会引入一些额外的运行时间开销；因此，只需放弃新提出的方法即可创造新的记录。因此，对于热门的竞争性任务，标准方法出现大幅但虚假的改进并持续出现在记录历史中的可能性很小。

举例来说，我将描述目前关于 Muon 的证据。它优于 AdamW 的主要证据来自其在“NanoGPT speedrunning”竞赛任务中的成功。具体来说，在 2024 年 10 月 15 日，从 AdamW 切换到 Muon 后，NanoGPT 的训练速度记录发生了翻天覆地的变化，Muon 将训练速度提高了 35%。此后，在 7 位不同的研究人员创造的 12 项新的 NanoGPT 速通记录中，Muon 一直是首选优化器。

Muon 的每步运行时间比 AdamW 长，因此，如果存在能够使 AdamW 的样本效率与 Muon 相当的超参数，那么只需放弃 Muon，重新启用 AdamW，就有可能创造新的记录。所以，要想相信 Muon 比 AdamW 更好（至少在训练小型语言模型方面），你其实完全不需要相信我（Keller Jordan）。相反，你只需要相信社区里有研究人员知道如何调优 AdamW，并且有兴趣创造新的 NanoGPT 速通纪录。这难道不美妙吗？

## Remaining open questions

- Muon能否扩展到更大规模的训练？（例如，1T 以上的 token，20B 以上的参数）
- 是否有可能将 Muon 使用的 Newton-Schulz 迭代次数合理地分布到大规模 GPU 集群上？
- Muon 是否只能用于预训练，而不能用于微调或强化学习工作负载？

截至撰写本文时，我尚不清楚这些问题的答案。

## Muon Contributors

以下研究人员对 Muon 项目做出了贡献。

- Jeremy Bernstein 和 Laker Newhouse 向我发送了他们的论文《旧优化器，新范式：文集》（Old Optimizer, New Norm: An Anthology），该论文在附录 A 中推荐使用牛顿-舒尔茨迭代法作为 Shampoo 的计算策略。在 Muon 开发和演示之前几个月，Jeremy 还在 X 论坛上发表了关于一种密切相关的算法——谱范数下的最速下降法的理论。最后，Jeremy 还指出，我早期版本的牛顿-舒尔茨迭代法的系数可以进一步调整。
- Vlado Boza 通过实验证明，Muon 分别应用于 Q、K、V 参数时效果更好，而不是将它们合并到一个矩阵中。
- Yuchen Jin 进行了实验，证明 Muon 训练可以扩展到更长的训练时间和更大的模型。他还为该项目提供了大部分必要的资金（以百小时计算）。
- Jeremy Bernstein、Jiacheng You 和 Franz Cesista 发现，我最初实现的 Newton-Schulz 迭代算法的效率可以从 6nm^ 提升到 $4nm^+2m^3$ FLOPs（对于形状为 $n\times m$ 的参数，其中 $m\le n$）。Jeremy Bernstein 和 Jiacheng You 同时发现了更优的变体，Franz Cesista 向速通代码库提交了 pull request，对变体进行了基准测试和实现。

## References
