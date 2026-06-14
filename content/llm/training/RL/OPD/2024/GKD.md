# ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES

论文链接：https://arxiv.org/pdf/2306.13649

代码链接：

## 摘要

知识蒸馏（KD）广泛用于压缩 teacher 模型，通过训练一个更小的 student 模型来降低其推理成本和内存占用。然而，目前针对自回归序列模型的 KD 方法存在一个问题：**训练过程中观察到的输出序列与学生在推理过程中生成的输出序列之间存在分布不匹配**。为了解决这个问题，我们提出了 Generalized Knowledge Distillation (GKD)。GKD 并非仅仅依赖于一组固定的输出序列，而是利用teacher 模型对这些序列的反馈，在 student 自身生成的输出序列上训练 student 模型。与监督式 KD 方法不同，GKD 还允许在 student 模型和 teacher 模型之间使用不同的损失函数，这在 student 模型缺乏足够的表达能力来模仿 teacher 模型的分布时非常有用。此外，GKD 还有助于将知识蒸馏与语言模型的强化学习微调无缝集成。我们证明了 GKD 在提炼自回归 T5 语言模型方面的有效性，包括针对摘要、翻译和推理任务的特定任务提炼，以及针对指令调整的任务无关提炼。

## 1.介绍

<img
  src="https://i-blog.csdnimg.cn/direct/dc8d12a2bd374d92b46541400a975e43.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

自回归序列模型，例如语言模型（LM），在众多任务中展现出了令人瞩目的能力，其成功的关键往往在于扩展训练数据量和模型参数数量。然而，扩展参数数量会带来成本，此类模型的部署会受到推理成本或内存占用的限制。因此，对于大型高性能模型的实际应用而言，一个至关重要的目标是在尽可能保持其性能的前提下，通过减少参数数量来压缩模型。

知识蒸馏是模型压缩的常用技术之一。知识蒸馏是指训练一个模型（student 模型）来复制另一个模型（teacher 模型）在特定任务集上的知识。通常，student 模型的参数比 teacher 模型少，因此，**知识蒸馏可以在保持比 teacher 模型更低的推理成本和内存占用的同时，提升特定任务的性能**。目前针对自回归序列模型的知识蒸馏方法要么需要 teacher 模型生成一组固定的输出序列（这可能很耗时），要么需要一个固定的序列数据集，teacher 模型可以通过分配 token 级概率来标记这些序列。然而，使用固定的数据集会导致训练期间观察到的输出序列与学生模型在推理期间自回归生成的序列之间的分布不匹配，这是模仿学习中一个众所周知的问题。此外，知识蒸馏的常见目标是最小化 teacher 模型和 student 模型分布之间的前向 KL 映射。**然而，student 的表达能力可能不足以符合 teacher 的分布，这可能导致 student 训练的样本不太可能由 teacher 生成**（例如，图 A.16）。

本文提出了一种 Generalized KD (GKD)  方法来缓解上述问题。首先，我们认识到自回归序列模型的知识分布可以看作是一个带有交互式专家的模仿学习问题。基于此，GKD 使用 student 模型自身生成的、on-policy 的序列（而非固定的输出序列集）来训练它，并将 teacher 概率作为这些序列的专家标签。我们的想法也得到了近期大语言模型在其自身输出序列上进行微调的成功案例的支持。此外，GKD 还提供了优化其他差异度量的灵活性，例如反向知蒸馏和广义 JSD（第 2 节），这些度量可以利用学生模型有限的学习能力，专注于生成 teacher 概率较高的样本。

GKD 整合了一些现有的自回归语言模型知识蒸馏 (KD) 方法，同时提出了新的 on-policy 方法，这些方法显著优于现有方法。就 on-policy GKD 相对于初始 student 模型的性能提升而言，在不同规模的 T5 student 模型上，我们观察到，与基线 KD 方法相比，GKD 在摘要任务上实现了 2.1 倍的相对提升，在机器翻译任务上实现了 1.7 倍的相对提升，在算术推理任务上实现了 1.9 倍的相对提升（图 1）。此外，我们还展示了 GKD 在与任务无关的知识蒸馏方面的有效性，在预留的 BBH 和 MMLU 基准测试套件上分别实现了 2% 和 1% 的绝对准确率提升（图 10）。

我们的主要贡献包括：
- 为了解决自回归语言模型训练和推理过程中出现的差异，我们提出了一种  on-policy 的学生生成输出进行知识蒸馏的方法，该方法以这些输出的 token 级教师概率为指导。在特定任务的知识蒸馏（图 1）和与任务无关的知识蒸馏（图 10）中，GKD 的性能均显著优于常用方法。
- 我们证明， on-policy GKD 可以与语言模型的 RL 微调（例如 RLAIF）无缝结合，这种组合以前从未被探索过（图 5）。
- 通过对 GKD 中的设计选择进行系统评估，我们提供了关于在蒸馏过程中使用 student 生成的 on-policy 输出序列的重要性以及 student 和 teacher 之间最优分歧的任务依赖性的实用见解。

## 2.PRELIMINARIES

**Auto-regressive Generative Sequence Models**。我们将输入序列和输出序列分别记为 $x$ 和 $y$。令 $\mathbb V$ 表示包含 $M$ 个 token 的词表，$y_{<n+1} = (y_1, y_2, ..., y_n)$ 表示生成的输出序列，直至第 $n$ 个 token，$L_y$ 表示序列 $y$ 的长度。token 级自回归策略 $p(·|y_{<n}, x) ∈ (0, 1)^M$ 输出 $\mathbb V$ 中所有 token 的下一个 token 概率分布，该分布以输入 $x$ 和输出序列 $y_{<n}$ 为条件。此外，$y ∼ p(·|x)$ 对应于给定输入 $x$ 的采样输出序列 $y$。为方便起见，我们定义 $p(y_n|x) := p(y_n|y_{<n}, x)$。自回归生成过程涉及基于先前生成的 token 逐个预测下一个 token。预测第 $n$ 个 token $y_n$ 的概率 $p(y_n|x)$ 由温度为 $γ$ 的 softmax 函数确定：$p(y_n|x) = \frac{exp(z_n/γ)}{\sum^M_{i=1}exp(z_i/γ)}$，其中 $z_n$ 是 token $y_n$ 的 logit 得分。$γ$ 值越高，随机性越强；$γ$ 值越低，输出越确定，因为 $γ$ 值越小，输出结果越接近最有可能出现的词。训练过程中，学生模型的温度始终保持在 1。评估时，我们使用贪婪采样 ($γ → 0$) 或温度采样 ($γ > 0$)。

**KL-Based Divergences**。两个概率分布之间的差异是衡量分布相似性的指标，其中 KL 散度是一种常用的指标。​​两个离散分布 $P(\mathcal C)$ 和 $Q(\mathcal C)$ 之间的 KL 散度由下式给出：$\mathcal D_{KL}(P∥Q) = \sum_{c∈\mathcal C} P(c) log \frac{P(c)}{Q(c)}$。

KL散度具有不对称性：$\mathcal D_{KL}(P∥Q) ≠ \mathcal D_{KL}(Q∥P)$。因此，我们将 $\mathcal D_{KL}(P∥Q)$ 称为 $P$ 和 $Q$ 之间的正向 KL 散度，而将 $\mathcal D_{KL}(Q∥P)$ 称为反向KL散度。在经验数据分布下，**正向 KL 散度对应于最大似然估计**，我们在监督学习中对其进行优化。当模型容量不匹配时，如果使用分布 $Q_θ(\mathcal C)$ 来近似 $P(\mathcal C)$，则最小化反向 KL 散度和正向 KL 散度会导致均值和众数搜索行为（图 A.16）。

尽管 KL 散度可能无界，但广义 JSD（Jensen-Shannon 散度）即使对于具有不相交支撑的概率分布也是有界的。$JSD(β)$ 使用有界系数 $0 < β < 1$ 在正向 KL 散度和反向 KL 散度之间进行插值：

```math
\mathcal D_{JSD(\beta)}=\beta\mathcal D_{KL}(P∥\beta P+(1-\beta)Q)+(1-\beta)\mathcal D_{KL}(Q∥\beta P+(1-\beta)Q)\tag{1}
```

Huszar´ (2015) 证明，$lim_{β→0} \mathcal D_{JSD(β)}(P∥Q)/β = \mathcal D_{KL}(P∥Q)$。因此，当 $β$ 分别接近 0 和 1 时，$JSD(β)$ 的梯度行为类似于正向 KL 和反向 KL。

## 3. DISTILLATION FOR AUTO-REGRESSIVE SEQUENCE MODELS

**Problem Setup**。我们给定两个容量不同的自回归序列模型，其中 $p_S$ 和 $p_T$ 分别代表 student 和 teacher。我们假设 student 具有可学习的参数 $θ$，并且 $p^θ_S$ 关于 $θ$ 可微。我们还给定一个输入数据集 $X$。此外，我们还可以假设可以访问一个输入-输出序列对 $(X, Y)$ 的数据集。如果没有给定，则可以通过从 teacher 中采样序列来生成这样的数据集。对于散度 $\mathcal D$，我们将 $p_T$ 和 $p_S$ 的 token 级分布之间的差异定义为

```math
\mathcal D(p_T∥p^{\theta}_S)(y|x):=\frac{1}{L_y}\sum^{L_y}_{n=1}\mathcal D(p_T(\cdot|y_{\lt n},x)∥p^{\theta}_S(\cdot|y_{\lt n,x})),\tag{2}
```

对于输入 $x$ 和输出序列 $y$。例如，在公式 2 中使用 $JSD(β)$ 作为 $\mathcal D$，得到 $\mathcal D_{JSD(β)}(p_T∥p^θ_S)(y|x) = \frac{1}{L_y}\sum_n D_{JSD(β)}(p_T(·|y_{<n}, x)∥p^θ_S(·|y_{<n}, x))$。

**Supervised FT**。如果我们只得到一个固定的真实输出序列数据集，但没有对 teacher 策略的查询访问权限，那么一个简单的方法是最小化学生策略下此类序列的负对数似然：$L_{SFT}(θ) = \mathbb E_{(x,y)∼(X,Y)}[−log p^θ_S(y|x)]$。

**Sequence-Level KD**。SeqKD 最大化 teacher 生成的高概率序列的可能性，可以看作是对教师生成的输出进行有监督式 FT。

**Supervised KD**。监督式 KD 是一种广泛使用的技术，它训练 student 模型模仿 teacher 模型的 token 级概率分布。student 模型 $p_S$ 使用监督式学习目标 $L_{SD}$ 进行训练，学习目标是 teacher 模型 $p_T$ 的目标 token 级概率分布：

```math
L_{SD}(\theta):=\mathbb E_{(x,y)\sim(X,Y)}[\mathcal D_{KL}(p_T||p^{\theta}_S)(y|x)],\tag{3}
```

其中期望值是基于数据集中的样本计算的。这种监督式目标函数利用 teacher 模型的完整 token 级分布，从而产生丰富的训练信号。

### 3.1 GENERALIZED KNOWLEDGE DISTILLATION (GKD)

<img
  src="https://i-blog.csdnimg.cn/direct/a3b676c1c76142a09fa480ed0bcd2f29.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

如上所述，常用的 KD 方法使用固定的输出序列数据集，可以是真实标签或 teacher 生成的序列。然而，使用此类方法蒸馏自回归 student 模型会导致训练集和推理集分布不匹配。这是因为 student 模型在推理阶段遇到的部分序列（自回归生成阶段）可能与训练阶段遇到的部分序列差异很大。由于自回归模型中每一步的预测都依赖于之前的步骤，这种不匹配会产生连锁反应，即早期步骤的预测误差会影响后续的预测，最终导致文本生成质量下降。为了解决这种不匹配问题，我们大量借鉴了模仿学习（IL）的思想。具体来说，**on-policy 模仿方法会迭代地使用 student 模型的策略收集序列，获取这些序列的专家标签，然后基于这些数据集重新训练 student 模型**。尽管 on-policy 的方法在机器人和深度强化学习领域应用广泛，但通常并不用于蒸馏自回归模型。

我们将 on-policy 模仿扩展到知识蒸馏，提出了 **on-policy KD**。在知识蒸馏过程中使用 on-policy 数据时，学习者会从 teacher 对自身生成的输出序列中错误 token 的logits中获得 token 级反馈。这形成了一种类似于强化学习（RL）中的反馈循环，有助于最小化训练-推理分布的不匹配。此外，随着学习者在训练过程中不断进化，其生成的数据质量也会提高。给定输入 $x$，学习者生成输出序列 $y$，并模仿 teacher 在中间状态 $y_{<n}$ 上的 token 级分布 $p_T(y_n|x)$。具体而言，on-policy 损失 $\mathcal L_{OD}$ 由下式给出：

```math
L_{OD}(\theta):=\mathbb E_{x\sim X}\left [\mathbb E_{y\sim p_S(\cdot|x)}[\mathcal D_{KL}(p_T||p^{\theta}_S)(y|x)]\right ],\tag{4}
```

我们不通过学生的采样分布 $p_S(·|x)$ 进行反向传播，这类似于 on-policy 模仿。不通过采样进行反向传播使得训练更加稳定且计算效率更高。在 on-policy KD 中，训练是在 student 可能生成的输出序列上进行的。训练过程中，我们使用温度 $γ = 1$ 来鼓励 student 生成的序列具有多样性。此外，给定未标记的输入提示，由于模型规模的差异，使用 student 生成序列比使用 teacher 生成序列的计算成本更低。

在 on-policy KD 方法的基础上，我们统一了有监督学习和 on-policy 的方法，并提出了一种更通用的方法，称为 Generalized KD (**GKD**)。**在 GKD 中，我们可以选择要优化的散度以及用于训练的输出序列**。具体来说，我们可以优化 teacher 和 student token 概率分布之间的任意散度。对于输出序列，GKD 使用固定数据集（teacher 生成或真实标签）和 on-policy 的学生生成序列的混合数据集。抽象地说，GKD 最小化如下形式的目标函数：

```math
\boxed{
L_{\mathrm{GKD}}(\theta)
:= (1-\lambda)\mathbb{E}_{(x,y)\sim (X,Y)}
\left[\mathcal{D}\left(p_T \middle\| p_S^\theta\right)(y|x)\right] + \lambda \mathbb{E}_{x\sim X}
\left[
\mathbb{E}_{y\sim p_S(\cdot|x)}
\left[
\mathcal{D}\left(p_T \middle\| p_S^\theta\right)(y|x)
\right]
\right]
}
```

其中 $\mathcal D(p_T, p_S)(y|x)$ 表示 teacher 分布和 student 分布之间的散度（公式 2），$λ ∈ [0, 1]$ 是一个超参数，用于控制 student 数据比例，即 on-policy 学生生成输出的比例。与 on-policy KD 类似，我们不通过学生的采样过程反向传播梯度。on-policy KD 和有监督 KD 分别是 GKD 的实例，其散度 $\mathcal D$ 设置为前向 KL，学生数据比例 $λ$ 分别设置为 1 和 0。也就是说，GKD 允许对比例 $λ$ 和散度进行其他选择，我们将在本文中对此进行探讨。

**Remark**。与随机初始化的 student 模型不同，我们假设可以使用一个能够生成足够高质量序列的 student 模型，teacher 可以对这些序列提供反馈。在我们的实验中，我们从经过有监督微调（FT）的 student 模型开始。这类似于两阶段 RLHF 训练，后者广泛应用于语言模型（LM）的训练，其中我们首先运行 SFT，然后进行 on-policy 强化学习（RL）微调。因此，GKD 可以利用 RLHF 的超参数调优经验，并且可以与 RLHF 结合使用，计算开销很小，无需额外的超参数。

**Choice of Divergence in GKD**。虽然前向 KL 算法常用于蒸馏，但它要求 student 模型覆盖 teacher 模型 token 级分布 $p_T(.|y_{<n}, x)$ 的整个支持域。这样做可能会导致学生模型将概率质量分配给在 $p_T(.|y_{<n}, x)$ 下概率较低的 token $v$，**从而产生幻觉和低质量的生成**。当 student 模型的容量远低于 teacher 模型时，使用温度采样时更容易出现这个问题（例如，图A.16）。另一种方法是寻找模式的散度算法，例如反向 KL 算法，它优先考虑 teacher 模型赋予高概率的 token，这可以避免低质量的生成，但代价是给定输入的生成多样性较低。我们的实验表明，最优发散算法似乎与任务相关。总而言之，在选择 GKD 散度算法时，需要考虑特定任务的多样性和性能之间的权衡（例如，图4、10）。

### 3.2 RL FINE-TUNING + ON-POLICY GKD

在某些任务中，从 teacher 模型中蒸馏可能只能提供主要目标的一个近似值，而该目标本身也可能是不可微的。我们可以直接使用强化学习（RL）来优化这个目标。方便的是，on-policy 的 GKD 可以轻松地与基于人类反馈（RLHF）或 AI 反馈（RLAIF）的强化学习微调相结合，因为它只需要 student 模型的输出样本。实际上，假设我们想要优化 student 模型的策略以获得标量奖励 $r$，同时保持其与 teacher 模型策略的接近，那么我们就可以得到如下形式的正则化强化学习微调目标：

```math
\mathbb{E}_{x \sim X}
\left[
(1-\alpha)
\underbrace{
\mathbb{E}_{y \sim p_S^\theta(\cdot|x)}[r(y)]
}_{\text{RL objective}} -
\alpha
\underbrace{
\mathbb{E}_{y \sim p_S(\cdot|x)}
\left[
\mathcal{D}\left(p_T \middle\| p_S^\theta\right)(y|x)
\right]
}_{\text{Generalized On-Policy Distillation}}
\right],
\tag{5}
```

其中 $α ∈ [0, 1]$ 控制蒸馏损失相对于强化学习目标函数的强度。当 α = 1 时，仅执行蒸馏操作。上述目标函数允许我们在最大化奖赏的同时，通过蒸馏提升模型的其他能力，**这或许可以减少在将语言模型与人类偏好对齐时，通用模型能力因“对齐税”而造成的下降**。我们将上述思想应用于 RLAIF 以缓解幻觉效应，同时通过蒸馏提升下游性能（图 5）。

**Remark**。在 RLHF 或 RLAIF 中，我们通常使用反向 KL 约束来限制学习到的策略，使其接近初始策略。如果只想对现有的强化学习微调流程进行少量修改，我们建议在将 GKD 与强化学习集成时使用反向 KL 或 JSD(0.9)。

## 4.EXPERIMENTS

### 4.1 CASE STUDY: ABSTRACTIVE SUMMARIZATION
