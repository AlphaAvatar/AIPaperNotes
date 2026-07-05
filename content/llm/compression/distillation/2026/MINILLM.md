# MiniLLM: On-Policy Distillation of Large Language Models

论文链接：https://arxiv.org/pdf/2306.08543

代码链接：https://github.com/microsoft/LMOps/tree/main/minillm

## 摘要

知识蒸馏（KD）是一种很有前景的技术，可以降低大语言模型（LLM）的高计算需求。然而，以往的 KD 方法主要应用于白盒分类模型或训练小型模型来模仿黑盒 API 模型，例如 ChatGPT。如何有效地将白盒 LLM 的知识蒸馏到小型模型中仍然是一个有待探索的问题，而随着开源 LLM 的蓬勃发展，这个问题变得愈发重要。**本文提出了一种将 LLM 蒸馏成小型语言模型的 KD 方法**。首先，我们将标准 KD 方法中的前向 Kullback-Leibler 散度（KLD）目标函数替换为更适合生成式语言模型的反向 KLD，以防止 student 模型高估 teacher 分布的低概率区域。然后，我们推导出一个有效的 on-policy 优化方法来学习该目标函数。student 模型被命名为 **MINILLM**。在指令遵循场景下的大量实验表明，与基线方法相比，MINILLM 能够生成更精确、整体质量更高、曝光偏差更小、校准效果更好、长文本生成性能更佳的响应。我们的方法可扩展至参数量在 120M 到 1.3B 之间的不同模型族。我们的代码、数据和模型检查点可在 https://github.com/microsoft/LMOps/tree/main/minillm 找到。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/e211ea32a9754ad2906fa4190a021a9a.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

随着大语言模型 LLM 的快速发展，知识蒸馏成为降低其高计算资源需求的一种常用技术。知识蒸馏是指利用大型 teacher 模型的监督信息来训练小型 student 模型。**知识蒸馏通常分为两类**：黑盒知识蒸馏和白盒知识蒸馏。黑盒知识蒸馏仅可访问 teacher 模型生成的文本，而白盒知识蒸馏则同时提供 teacher 模型的输出分布或中间隐藏状态。近年来，黑盒知识蒸馏在利用 LLM API 生成的提示-响应对微调小型模型方面展现出了良好的效果。随着更多开源 LLM 的出现，白盒知识蒸馏（KD）对研究界和产业界都变得更有价值，因为 student 模型可以从 teacher 模型的输出分布和隐藏状态中接收到更好的信号，从而有可能获得更高的性能。然而，白盒 KD 方法的研究主要集中在小型（<1B 参数）语言理解模型，而针对 LLM 的白盒 KD 仍有待探索。

本文研究了基于已知 teacher 模型输出分布的 LLM 的白盒知识蒸馏（KD）问题。我们认为，对于以生成方式执行任务的 LLM，标准的 KD 目标函数并非最优。给定 teacher 分布 $p(y|x)$ 和由参数 $θ$ 参数化的 student 分布 $q_θ(y|x)$，标准的 KD 目标函数（包括一些针对序列级模型的变体）本质上是最小化 teacher 分布和 student 分布之间的近似前向 Kullback-Leibler 散度（KLD），记为$KL[p||q_θ]$，该散度使得 $q_θ$ 覆盖 $p$ 的所有众数。对于文本分类任务，$KL[p||q_θ]$ 效果良好，因为输出空间通常由有限数量的类别组成，使得 $p(y|x)$ 和  $q_θ(y|x)$ 的众数都很少。然而，对于开放式文本生成任务（通常是 LLM 应用的情况），输出空间要复杂得多，由于模型容量有限，$p(y|x)$ 可以包含比 $q_θ(y|x)$ 所能表达的更多的模式。最小化前向 KLD 会导致 $q_θ$ 为 $p$ 的空白区域赋予过高的概率，并在自由运行生成过程中产生 $p$ 下非常不可能的样本。

为了缓解这个问题，我们提出最小化反向 KLD，即 $KL[q_θ||p]$，它广泛应用于计算机视觉和强化学习。与 $KL[p||q_θ]$ 相比，最小化 $KL[q_θ||p]$ 使得 $q_θ$ 倾向于寻找 $p$ 的主要模式，并为 $p$ 的空区域赋予较低的概率，如图 2 所示，并在 2.1 节中进行了讨论。在文本生成中，这意味着学习者避免学习 teacher 分布中过多的长尾变体，从而专注于生成的正确性，这在需要真实性和可靠性的实际场景中至关重要。为了优化 $min_θ KL[q_θ||p]$，如 2.2 节所示，我们使用策略梯度推导其梯度，并采用 on-policy 的训练方法。为了进一步稳定和加速训练，我们提出了以下方法：（1）单步分解以降低方差；（2）teacher 混合采样以缓解奖赏操纵；（3）长度归一化以消除长度偏差。最后，我们在第 2.3 节中介绍了整体的 KD 算法。我们的 student 模型被命名为 **MINILLM**，表明我们的方法适用于压缩大型（生成式）语言模型。

我们将该方法应用于各种生成式语言模型，这些模型的规模从 120M 到 1.3B 不等，均采用指令遵循场景，涵盖了广泛的自然语言处理任务。我们使用 5 个数据集，包括 Rouge-L、GPT-4 反馈和人工评判进行评估。实验表明，MINILLM 在所有数据集上均优于标准 KD 基线方法，并且能够很好地从 120M 模型扩展到 1.3B 模型（见图 1）。进一步分析表明，MINILLM 能够降低暴露偏差，提高校准效果，并提升长响应生成性能，同时将多样性损失忽略不计。

<img
  src="https://i-blog.csdnimg.cn/direct/e206cc43d99345efa6b12024f52d48b7.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

## 2.Method

<img
  src="https://i-blog.csdnimg.cn/direct/e89be95d9ded48488a2bf882b74ea4b0.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们考虑条件文本生成，其中模型根据从分布 $p_x$ 中采样的提示 $x$ 生成响应 $y = \{y_t\}^T_{t=1}$，这通常是 LLM 执行任务的方式。

**我们将知识蒸馏（KD）问题建模为一个优化问题**，旨在最小化固定的 teacher 模型分布 $p(y|x)$ 与由参数 $θ$ 定义的 student 模型分布 $q_θ(y|x)$ 之间的差异。标准的 KD 方法近似地最小化前向KL 损失（KLD）：$KL[p||q_θ] = \mathbb E_{x∼p_x,y∼p′} log \frac{p(y|x)}{q_θ(y|x)}$，其中 $p′$ 可以是真实数据分布（word-level KD）或 teacher 分布 $p$（sequence-level KD）。尽管 $KL[p||q_θ]$ 应用广泛，但当 $q_θ$ 的表达能力不足时，它往往会高估文本生成任务中 $p$ 的空白区域。LLM 的 KD 方法会遇到这种情况，因为 LLM 以生成的方式执行任务，其低容量的 student 模型无法完美地模仿 teacher 模型或人类复杂的文本生成分布。

### 2.1 MINILLM: Knowledge Distillation with Reverse KLD

我们将最小化 student 模型分布和 teacher 模型分布之间的反向 KLD 作为 MINILLM 的学习目标：

```math
\begin{aligned}
\theta=arg\mathop{min}\limits_{\theta}\mathcal L(\theta)
&=arg\mathop{min}\limits_{\theta} KL[q_\theta||p]\\
&=arg\mathop{min}\limits_{\theta} \left[-\mathbb{E}_{x\sim p_x,y\sim q_{\theta}}log\frac{p(\textbf y|\textbf x)}{q_{\theta}(\textbf y|\textbf x)}\right].
\end{aligned}\tag{1}
```

研究表明，最小化反向 KLD 会导致生成模型中的模式搜索行为，其中 $q_θ$ 会为 $p$ 的大模式赋予高概率，而忽略小模式（如图 2 所示的简单实验）。本文首先研究了文本生成中 LLM 的 KD 的这一特性。最小化正向 KLD 会导致 $q_θ$ 将较大的概率质量赋予 $p$ 的零概率区域，这在实践中对应于低质量文本的生成；而反向 KLD 则关注 $p$ 的主要模式，这对于确保文本生成的正确性和准确性至关重要。如图 3 所示，与最小化正向 KLD 的序列级 KD 不同，最小化反向 KLD 的 MINILLM 不会强制 $q_θ$ 拟合从 teacher 分布 $p$ 中采样的所有 $y$。相反，它鼓励 student 在其自身能力范围内生成 teacher 偏好的样本，这更容易实现。有趣的是，我们还从逆强化学习的角度找到了理解 MINILLM 的另一个视角。相关推导过程见附录 A.1。

### 2.2 On-Policy Distillation

**Gradient Derivation**。我们注意到，方程（1）中目标函数 $\mathcal L(θ)$ 的梯度可以利用策略梯度定理（Policy Gradient Theorem）推导得出，用于 on-policy 优化：

```math
\nabla\mathcal L(\theta)=-\mathbb{E}_{x\sim p_x,y\sim q_{\theta}(\cdot|x)}\sum^T_{t=1}(R_t-1)\nabla log~q_{\theta}(y_t|\textbf y_{\lt t},\textbf x),\tag{2}
```

其中 $T = |\textbf y|$ 以及 $R_t = \sum^T_{t′ = t}log \frac{p(y_{t′} | y_{< t′},\textbf x)}{q_θ(y_{t′} | y_{< t′}, x)}$ 是 $r_{t′} = log \frac{p(y_{t′}| y_{<t′}, x)}{q_θ(y_{t′} | y_{<t′} , x)}$ 的累积值，用于衡量每一步生成的质量。**直观地说，生成的文本应该通过提高 $p(y_{t′} | y_{<t′} , x)$ 来在 teacher 分布下具有较高的概率，同时通过降低 $q_θ(y_{t′} | y_{<t′} , x)$ 来保持多样性**。公式 2 中的期望值通过蒙特卡罗抽样计算。完整的推导过程见附录 A.2。然而，尽管有一些后续的解决方案，策略梯度仍然存在方差过大和奖励作弊的问题。此外，**我们注意到 $R_t$ 倾向于短句，这会导致 student 模型输出空响应**。因此，我们提出了三种策略来缓解这些问题。

**Single-Step Decomposition**。[CPO+19] 发现单步生成质量 $r_t$ 对训练方差至关重要，因为前几个 token 的误差会沿着整个句子累积。为了更关注 $r_t$，我们重写 $∇\mathcal L(θ)$ 以将 $R_t$ 分解为 $r_t$，并直接计算 $\mathbb E_{y_t∼q_θ(t)}[r_t]$ 的梯度（完整推导见附录 A.3）：

```math
\begin{aligned}
\nabla\mathcal L(\theta)
&=\mathbb E_{\textbf x\sim p_x,\textbf y\sim q_{\theta}(\cdot|x)}\left[-\sum^T_{t=1}\nabla\mathbb E_{y_t\sim q_{\theta}(t)}[r_t]\right] + \mathbb E_{\textbf x\sim p_x,\textbf y\sim q_{\theta}(\cdot|x)}\left[-\sum^T_{t=1}R_{t+1}\nabla log~q_{\theta}(y_t|\textbf y_{\lt t},\textbf x)\right]\\
&=(\nabla\mathcal L)_{single}+(\nabla\mathcal L)_{long},
\end{aligned}\tag{3}
```

其中 $q_θ(t) = q_θ(·|y_{<t}, x)$。注意，$\mathbb E_{y_t∼q_θ(t)}[r_t]$ 可以直接通过对词表求和来计算，而无需使用蒙特卡罗采样，并且可对 $θ$ 求导。这种分解方法能够更精确、更高效地估计单步生成质量，从而降低训练过程中的方差并加速收敛。

**Teacher-Mixed Sampling**。我们观察到，在使用公式 2 进行训练时存在奖赏作弊现象，因为 $q_θ$ 有时会产生退化的句子 $y$，这些句子在采样过程中会从 teacher 那里获得高分（例如，重复的短语），尤其是在 student 模型规模较小的情况下。为了创建更好的采样分布，我们在每个时间步混合 teacher 分布和 student 分布：

```math
\tilde p(y_t|\textbf y_{\lt t},\textbf x)=\alpha\cdot p(y_t|\textbf y_{\lt t}, \textbf x)+(1-\alpha)\cdot q_{\theta}(y_t|\textbf y_{\lt},\textbf x),\tag{4}
```

其中 $α$ 控制 teacher 混合的强度。从 $\tilde p$ 中采样可以抑制 teacher 辅助下的低质量生成，并缓解奖赏作弊。我们使用重要性采样重写 $(∇\mathcal L)_{Single}$ 和 $(∇_L)_{Long}$，以获得梯度的无偏估计：

```math
\begin{array}{cc}
(∇\mathcal L)_{Single}=-\mathbb E_{x\sim p_x,\textbf y\sim\tilde p(\cdot|\textbf x)}\left[\sum^T_{t=1}w_t\nabla\mathbb E_{y_t\sim q_{\theta}(t)}[r_t]\right],\\
(\nabla\mathcal L)_{Long}=-\mathbb E_{x\sim p_x,\textbf y\sim\tilde p(\cdot|\textbf x)}\left[\sum^T_{t=1}w_tR_{t+1}\nabla log~q_{\theta}(y_t|\textbf y_{\lt t},\textbf x)\right],
\end{array}\tag{5}
```

其中 $w_t =\prod^t_{t′=1}\frac{q_θ(y_{t′} |\textbf y_{<t′},\textbf x)}{\tilde p(y_{t′} |\textbf y_{<t′} ,\textbf x)}$ 是重要性权重。然而，由于 $w_t$ 需要在多个时间步长内将每个 token 的重要性权重相乘，因此在实践中会导致较高的方差，每个时间步长的方差都会累积。因此，我们近似地将 $wt$ 设置为 $w_t≈\frac{q_θ(y_t|\textbf y_{<t},\textbf x)}{\tilde p(y_t|\textbf y_{<t},\textbf x)}$，以降低公式 5 中估计器的方差。

**Length Normalization**。我们发现，长序列往往具有较小的 $R_{t+1}$ 值，这会导致模型产生较短的响应。因此，我们在公式 3 中对 $R_{t+1}$ 进行了长度归一化：


```math
R^{Norm}_{t+1}=\frac{1}{T-t-1}\sum^T_{t'=t+1}log\frac{p(y_{t'}|\textbf y_{<t'},\textbf x)}{q_{\theta}(y_{t'}|\textbf y_{<t'},\textbf x)}.\tag{6}
```

**In Summary**。结合以上策略，我们得到最终的优化梯度：

```math
\nabla \mathcal{L}(\theta)
= - \mathbb{E}_{\substack{x \sim p_x \\ y \sim \tilde{p}(\cdot \mid x)}}
\left[
\sum_{t=1}^{T}
w_t
\left[
\underbrace{
\nabla
\sum_{y' \in V}
q_\theta(y' \mid y_{<t}, x)
\log
\frac{
p(y' \mid y_{<t}, x)
}{
q_\theta(y' \mid y_{<t}, x)
}
}_{(\nabla \mathcal{L})_{\mathrm{Single}}\ \mathrm{part}}
+
\underbrace{
R_{t+1}^{\mathrm{Norm}}
\frac{
\nabla q_\theta(y_t \mid y_{<t}, x)
}{
q_\theta(y_t \mid y_{<t}, x)
}
}_{(\nabla \mathcal{L})_{\mathrm{Long}}^{\mathrm{Norm}}\ \mathrm{part}}
\right]
\right].
```

其中 $V$ 是语言模型的词表大小，$(∇\mathcal L)^{Norm}_{Long}$ 是 $R^{Norm}_{t+1}$ 的 $(∇\mathcal L)_{Long}$。

### 2.3 Training Algorithm

<img
  src="https://i-blog.csdnimg.cn/direct/fcbb349539904a1ea2c0fad06e7e44ce.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们从一个在大型长文档语料库 $\mathcal D_{PT}$ 上预训练的 student 模型开始。算法 2.3 通过将学生模型适配到使用数据集 $\mathcal D$ 的文本生成任务，并结合 teacher 模型（例如在 $\mathcal D$ 上微调的 LLM 或具有良好任务泛化能力的 LLM）的监督，来训练 MINILLM。在训练算法中，我们首先在 $\mathcal D$ 上微调 student 模型，并选择损失最低的检查点作为后续训练的初始化。然后，我们基于公式 5 和公式 6 计算梯度 $(∇\mathcal L)_{Single}$ 和 $(∇\mathcal L)^{Norm}_{Long}$，并添加裁剪策略 [SWD+17] 以进一步提高稳定性。与 [OWJ+22] 相同，我们引入语言建模损失 $\mathcal L_{PT} = − \mathbb E_{d∼\mathcal D_{PT}} log~q_θ(\textbf d)$ 以保持模型在标准 NLP 基准测试上的性能。最后，使用梯度组合 $(∇\mathcal L)_{Single} + (∇\mathcal L)^{Norm}_{Long} + ∇\mathcal L_{PT}$ 对 student 模型进行更新。整个基于策略的训练流程类似于基于人类反馈的强化学习。

## 3.Experiments

### 3.1 Experimental Setup

### 3.2 Results
