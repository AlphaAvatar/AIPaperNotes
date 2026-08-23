# LOOPED TRANSFORMERS ARE BETTER AT LEARNING LEARNING ALGORITHMS

论文链接：https://arxiv.org/pdf/2311.12424

代码链接：https://github.com/Leiay/looped_transformer

## 摘要

Garg et al. (2022) 的研究表明，Transformer 模型在解决各种（潜在）模型的上下文数据拟合问题方面展现出了有效性。然而，Transformer 架构本身缺乏迭代结构，这给模拟传统机器学习方法中常用的迭代算法带来了挑战。为了解决这个问题，我们提出了一种 **Looped Transformer** 架构及其相关的训练方法，旨在将迭代特性融入 Transformer 架构中。实验结果表明，循环 Transformer 在解决各种数据拟合问题时，性能与标准 Transformer 相当，而参数数量却减少了不到 10%。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/0b5181b7dd2c45cdadcd835a6873c79a.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

Transformer 模型已成为自然语言处理（NLP）及其他需要序列到序列建模领域的首选模型。除了在自然语言处理任务中展现出卓越的性能外，诸如 GPT-3 和 PaLM 等大语言模型（LLM）还展现出上下文学习能力：它们能够根据简短的提示适应各种下游任务，从而无需进行额外的模型微调。这种引人入胜的上下文学习能力激发了研究界的浓厚兴趣，并催生了大量相关研究。然而，**Transformer 模型实现上下文学习的底层机制仍不甚明了**。

为了理解 LLM 的上下文学习行为，Garg et al. (2022) 研究了从零开始训练的 Transformer 模型在解决特定函数类上下文学习问题时的性能。值得注意的是，Transformer 模型在所有任务中都表现出色，其性能与传统求解器相当甚至更优。在此基础上，Akyurek et al. (2022) 探索了基于 Transformer 的模型解决线性回归学习问题的能力，并将其解释为现有学习算法的一种隐式形式。他们的研究结合了理论和实证视角，以理解 Transformer 模型如何学习这些函数。随后，von Oswald et al. (2022) 通过实证证明，当训练预测线性函数输出时，仅使用线性自注意力机制的 Transformer 模型能够自然地学习执行单步梯度下降来解决上下文中的线性回归任务。尽管 von Oswald et al. (2022) 提出的方法和基础理论的研究成果令人鼓舞，但他们所考察的简化架构与实践中使用的标准解码器 Transformer 之间仍存在显著差距。如何仅通过少量架构修改从头开始训练一个标准解码器 Transformer，以有效复现学习算法，仍然是一个悬而未决的问题。

在传统机器学习中，迭代算法常用于求解线性回归问题。然而，标准 Transformer 模型所采用的方法并非天然适合迭代计算。Looped Transformer 架构（例如 Giannou et al. (2023) 的研究）为弥合这一差距提供了一条很有前景的途径。**循环 Transformer 架构不仅具有以迭代方式解决问题的固有优势，还能将任务分解为更简单的子任务，从而可能显著减少模型参数**。

为了说明任务分解如何节省参数，我们仔细研究一下如何使用 Transformer 模型来解决线性回归任务，特别是 $min_w ∥\textbf X\textbf w − \textbf y∥^2_2$ 中的 $\textbf w$（图 1）。为了训练 Transformer 模型完成此任务，我们输入一个格式为 $(\textbf x_1, \textbf w^T\textbf x_1, ..., \textbf x_k, \textbf w^T\textbf x_k, \textbf x_{test})$ 的提示序列。其中，$\textbf w$ 代表线性回归模型的参数，$\{\textbf x_1, ..., \textbf x_k\}$ 是 $k$ 个上下文样本，$\textbf x_{test}$ 是测试样本。Transformer 可以通过单次前向传播来尝试预测 $\textbf y_{test}$，方法是近似普通最小二乘法解。与矩阵乘法相比，Transformer 更难学习计算普通最小二乘法解 $(\textbf X^T\textbf  X)^{−1}\textbf X^T\textbf y$ 中所需的矩阵逆。这是因为逆运算需要更多的层和头。然而，梯度下降法为线性回归提供了一种替代解决方案，它只需要矩阵乘法：$\textbf X^T(\textbf X\textbf w − \textbf y)$，但需要重复应用。

基于此观察，我们提出以下问题：

> **Looped Transformer 能否比标准的非递归 Transformer 更有效地模拟迭代学习算法？**

在我们测试的特定函数类中，答案是肯定的。图 2 展示了我们使用 Looped Transformer 解决线性回归任务的初步结果。本文的其余部分组织如下：在第 4 节中，我们开发了一种循环 Transformer 的训练方法，以模拟迭代算法的预期性能。随后，在第 5 节中，我们比较了标准 Transformer 和循环 Transformer 的实验性能，并分析了它们之间的权衡。我们的贡献和发现概述如下：

**Training methodology for Looped Transformer**。我们提出了一种循环 Transformer 的训练方法，旨在有效地模拟迭代算法。对于模拟收敛算法的 Looped Transformer，其假设是随着循环迭代次数的增加，其性能应得到提升或最终收敛。基于此假设，我们深入研究了 Looped Transformer 的结构设计，并探讨了训练过程中所需的循环迭代次数。这些研究最终促成了我们训练方法的提出。

**Performance of Looped Transformers for in-context Learning**。基于我们提出的训练算法，实证研究表明，Looped Transformer 可以从零开始训练，从而在上下文中学习由线性函数、稀疏线性函数、决策树和两层神经网络生成的数据。在所考察的各种函数类型中，循环 Transformer 的性能始终优于标准 Transformer，尤其是在数据由稀疏线性函数或决策树生成时。我们的研究结果表明，循环 Transformer 在上下文中模拟学习算法方面可能更有效，尤其适用于本文探讨的学习任务。

<img
  src="https://i-blog.csdnimg.cn/direct/d1e2f1f36c9d40bb85ba4a8ca2d06fc0.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## 2.RELATED WORKS

**Weight Sharing Models**。权重共享是循环神经网络（RNN）和长短期记忆网络（LSTM）等循环模型固有的特性。这些序列模型在序列处理过程中复用权重，从而能够处理长度几乎无限的序列输入。Transformer 模型通常具有固定的上下文长度。为了在不增加参数的情况下适应更长的上下文，一些研究工作将 Transformer 应用于循环框架中，以实现自适应前向传播或扩展上下文长度。此外，一些研究工作还提出了在模型前向传播过程中进行权重共享的方法。这种权重共享要么仅限于有限数量的层，要么通过设置终止准则在必要时退出循环。一种极端情况是隐式模型，其中函数被无限次重复应用。下一段将讨论这些隐式模型的应用。

**Deep Implicit Model**。深度隐式模型采用黑盒求解器来寻找隐式深度模型的不动点解。随后， Later, Bai et al. (2021)  提出了雅可比矩阵正则化来稳定训练过程。然而，这种方法需要大量的超参数调优，并且仍然存在不稳定性问题。隐式模型已被证明能够解决具有外推能力的数学问题。这是使用循环模型解决数学任务的一个特例，我们将在下一段中更详细地探讨。

**Using Transformer to Solve Math or Reasoning Tasks**。已有若干研究探索了深度神经网络（包括循环模型）在解决涉及迭代过程的数学或推理任务方面的能力。这些任务涉及有限的状态或选择。此外，Zhang et al. (2023b) 研究了 Transformer 如何模拟结构递归问题，而 Ongie et al. (2020) 则研究了递归模型在解决图像处理逆问题中的应用。另一方面，回归问题涉及连续的输出空间。Garg et al. (2022) 研究了 Transformer 学习连续函数类的能力。Akyurek et al. (2022) 研究了不同深度 Transformer 在训练线性回归和噪声线性回归问题时所获得解的相变。Raventos et al. (2023) 研究了训练过程中遇到的任务数量与 Transformer 在上下文中学习岭回归问题的性能之间的关系。然而，我们的工作采用了与 Garg et al. (2022) 相同的架构 —— 一种具有非线性注意力的解码器模型——旨在提供关于该模型学习动态的经验证据和见解。

**Understanding How Transformer Learns In-Context to Solve Data-Fitting Problem**。除了从经验角度研究这个问题之外，一些研究也从理论角度对其进行了考察，重点关注 Transformer 模型和线性注意力网络（不包括 softmax）的构建。他们将 Transformer 模型在线性回归任务中的可学习性与 query 提示预测的隐式梯度下降更新联系起来。von Oswald et al. (2022) 通过实证研究表明，在最小化预测值与真实标签之间的 $ℓ_2$ 距离时，单层线性自注意力（LSA）Transformer 模型在解决线性回归问题时，梯度下降一步是最优解；而 Ahn et al. (2023); Zhang et al. (2023a; 2024) and Mahankali et al. (2023) 则对此给出了理论解释。除了单层 LSA 模型之外，Fu et al. (2023); Giannou et al. (2024) 的研究也支持这一观点，其研究表明多层 Transformer 模型可以执行高阶优化。此外，另一项研究从贝叶斯估计的角度对 Transformer 模型的上下文学习能力进行了理论阐释。 Li et al. (2023) 从泛化误差的角度研究了上下文学习。Lin & Lee (2024) 研究了上下文学习的双重运行模式。

## 3.PROBLEM SETTING

令 $\mathcal F$ 表示定义在 $\mathbb R^d$ 上的一类函数。令 $\mathcal D_{\mathcal F}$ 表示 $\mathcal F$ 上的概率分布，$\mathcal D_{\mathcal X}$ 表示 $\mathbb R^d$ 上的概率分布。随机学习提示 $P$ 的生成方式如下：从 $\mathcal D_{\mathcal F}$ 中采样函数 $f$，从 $\mathcal D_{\mathcal X}$ 中采样输入 $\{x_1, · · · , x_k\}$ 以及测试样本 $x_{test}$。输出 $x_i$ 由 $f(x_i)$ 计算得出。提示 $P$ 为 $P = (x_1, f(x_1), · · · , x_k, f(x_k), x_{test})$，学习系统的目标是预测 $f(x_{test})$。令 $M$ 为一个学习系统，$M(P)$ 表示其预测结果（注意，预测结果中并未显式给出 $f$）。$M$ 的性能由平方误差 $ℓ(M(P), f(x_{test})) = (M(P) − f(x_{test}))2$ 来衡量。本文重点研究基于 Transformer 的学习系统，并根据不同的任务将其与其他已知的学习系统进行比较。具体来说，我们研究了具有 $L$ 层的 GPT-2 解码器模型。默认情况下，非循环 Transformer 的 $L$ 值为 12，遵循 Garg et al. (2022) 的设置；Looped Transformer 的 $L$ 值为 1。

## 4.TRAINING ALGORITHM FOR LOOPED TRANSFORMER

本节将深入探讨算法模拟 looped transformer 的设计选择。对于算法模拟 looped transformer，我们预期其具备以下特性：1）随着循环迭代的进行，looped transformer 的性能应保持或提升；2）循环迭代有可能无限期地持续进行而不会导致性能下降。

**Training Strategy**。基于第 3 节的问题设定，令 Transformer 的提示为 $P = (x_1, f(x_1), · · · , x_k, f(x_k), x_{test})$，其中 $P^i$ 表示带有 $i$ 个上下文样本的提示前缀，即 $P^i = (x_1, f(x_1), · · · , x_i, f(x_i), x_{i+1})$。Looped Transformer 经过 $t$ 次循环迭代后的输出为

```math
Y_t(P^i|\theta)=\underbrace{M_{\theta}(M_{\theta}(\cdots M_{\theta}(}_{\text{t iterations}}Y^i_0+P^i)+P^i)\cdots + P^i)
```

其中，Transformer $M$ 由 $θ$ 参数化，$Y^i_0$ 是一个与 $P^i$ 形状相同的零张量。然后，我们通过最小化以下期望损失来训练 Transformer：

```math
\mathop{min}\limits_{\theta}\mathbb E_P\bigg[\frac{1}{b-b_0}\sum^b_{t=b_0}\frac{1}{k+1}\sum^k_{i=0}(Y_t(P^i|\theta),f(\textbf x_{i+1}))\bigg],\tag{1}
```

我们仅测量循环迭代次数从 $b_0$ 到 $b$ 的所有提示前缀的 transformer 损失，其中 $b_0 = max(b − T, 0)$。这种截断损失的灵感来源于时间截断反向传播算法，旨在节省计算量。

**Model Configuration and Parameter Count**。为了与非循环 Transformer 进行比较，我们使用了一个与 Garg et al. (2022) 描述的 Transformer 完全相同的模型，只是层数不同。具体来说，我们采用了一个 GPT-2 模型，其嵌入维度为 $D = 256$，注意力头数为 $h = 8$。标准（非循环） Transformer 有 $L = 12$ 层，而循环 Transformer 有 $L = 1$ 层。在参数数量方面，非循环 Transformer 包含 9.48M 参数，而循环 Transformer 使用 0.79M 参数。我们采用相同的训练策略：使用 Adam 优化器，学习率为 0.0001，不使用权重衰减或其他显式正则化（例如梯度裁剪或数据增强）。

**Key Factors for Finding a Fixed-point Solution**。在部署这种循环架构训练时，需要考虑两个关键因素：1）循环架构中的输入注入方式（第 4.1 节），以及 2）训练期间的最大循环迭代次数（第 4.2 节）。后续章节将以线性回归作为上下文学习的具体任务，阐述这两个因素的影响。

### 4.1 INPUT INJECTION TO LOOPED TRANSFORMER

<img
  src="https://i-blog.csdnimg.cn/direct/d841894fdf4c48a7a3a9e108c122994a.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

沿用一些符号，令 $P$ 表示 Transformer 模型 $M(·)$ 的输入，$Y_t$ 表示应用 $M(·)$ $t$ 次迭代后的输出。一般而言，循环 Transformer 可以表示为 $Y_{t+1} = M(Y_t; P), ∀t$。一些研究探讨了一种称为权重绑定形式的特定变体：$Y_{t+1} = M(Y_t)$，其中 $Y_0 = P$，且对于某个常数 $T$，$t < T < ∞$。

然而，当 $t$ 趋于无穷大时，初始输入 $Y_0$ 的影响会减弱，解本质上变得随机或不可预测。为了将输入 $P$ 融入循环 Transformer 的解中，我们建议设置 $Y_{t+1} = M(Y_t + P)$。需要注意的是，输入注入方法并非仅限于加法。图 3 展示了使用（默认）输入注入或不使用（称为权重绑定）输入注入训练循环 Transformer 的结果。在推理过程中，权重绑定 Transformer 在训练循环迭代后会迅速发散。

### 4.2 CHOICE OF LOOP ITERATIONS

<img
  src="https://i-blog.csdnimg.cn/direct/5b32d54e2b424476b44b90c74d8e9b2d.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

为了评估循环 Transformer 模型在第 $b$ 次循环迭代时的性能，我们需要连续执行 $b$ 次计算。这可以类比为一个实际上拥有 $b$ 层的 Transformer 架构，尽管这些层之间共享权重。选择 $b$ 值需要权衡。较高的 $b$ 值会导致更长的训练和推理时间，而较低的 $b$ 值则会限制模型的潜力。此外，如公式 1 所示，损失函数被截断，只有 $T$ 次循环迭代的输出才会影响损失函数。

图 4 表明，较低的 $b$ 值可能导致循环 Transformer 在训练循环迭代后发散，从而得到鲁棒性较差的不动点解。另一方面，较高的 b 值则能使模型找到一个在训练循环迭代后不会发散的不动点解。然而，超过某个 $b$ 值后，循环迭代的收敛速度会下降。

类似的权衡也适用于 $T$ 的选择；较低的 $T$ 值可能会降低性能，而较高的 $T$ 值可能会导致训练不稳定，这种现象在循环模型训练文献中已有记载。$b = 12$ 的循环 Transformer 与 Garg et al. (2022) 研究的非循环 Transformer 的有效深度相匹配。然而，它未能复现标准（非循环）Transformer 的性能。

另一个观察结果是，循环 Transformer 总能找到一个不动点解，该解在训练迭代次数 $b$ 之前就会达到饱和。不动点解的饱和是由于损失函数的要求，即循环 Transformer 需要在 $b$ 步内达到目标值。因此，选择较小的 $b$ 值可以加快收敛到不动点解的速度。然而，当 $b$ 值超过某个阈值后，无论 $b$ 值如何增加，收敛速度都会达到饱和。例如，使用 $T = 15, b = 40$ 和 $50$ 进行训练，得到的收敛速度相近。

**Optimal Choice of b and T**。我们的目标是高效地训练用于非均匀分布任务的循环 Transformer 模型。这需要确定最小的 $b$ 值，以防止模型在训练循环迭代后发散。为了最大限度地减少内存使用，我们倾向于较小的 $T$ 值。基于这些标准，我们为线性回归任务选择 $b = 20$ 和 $T = 15$。

## 5.EXPERIMENTAL RESULTS

### 5.1  EXPERIMENTAL SETUP

### 5.2 LOOPED TRANSFORMER CAN IN-CONTEXT LEARN LINEAR REGRESSION
