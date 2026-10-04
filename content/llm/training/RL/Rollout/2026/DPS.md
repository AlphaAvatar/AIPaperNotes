# DYNAMICS-PREDICTIVE SAMPLING FOR ACTIVE RL FINETUNING OF LARGE REASONING MODELS

论文链接：https://arxiv.org/pdf/2603.10887

代码链接：https://github.com/maoyixiu/DPS

## 摘要

强化学习（RL）微调已成为提升大语言模型（LLM）推理能力的关键技术，但其成效在很大程度上取决于训练数据的选择。**近期的研究凸显了在线提示词（prompt）选择方法的重要性**；这些方法通常侧重于针对当前策略下“部分已解决”或“难度适中”的样本进行训练，从而实现更有效的模型更新。尽管这些方法能显著减少RL微调所需的训练步数，但它们也带来了巨大的计算开销——为了筛选出具有信息量的样本，往往需要在大量候选提示词上进行密集的LLM推理采样（rollout），这种开销甚至可能超过微调过程本身的成本。为应对这一挑战，本文提出了 **Dynamics-Predictive Sampling (DPS)** 方法。该方法通过在进行高成本 rollout 之前推断提示词的学习动态，实现在线预测并筛选出具有信息量的提示词。具体而言，我们引入了一种新视角，**将 RL 微调过程中每个提示词的求解进度建模为一个动态系统**：其中求解程度表示为系统状态，而状态转移则由隐马尔可夫模型（HMM）刻画。利用历史采样获得的奖赏信号，我们执行在线贝叶斯推断以估计动态变化的状态分布；推断结果为高效的提示词选择提供了预测性先验，从而避免了依赖大量采样的筛选过程。在数学、规划和视觉几何等多种推理任务上的实验结果表明，DPS能显著减少冗余采样，加速训练过程，并实现更优的推理性能。相关代码已开源至：https://github.com/maoyixiu/DPS。

## 1.INTRODUCTION

强化学习（RL）微调已成为增强大语言模型（LLM）推理能力的关键技术。这些经过微调的模型通常被称为“大型推理模型”（LRM），它们通过生成思维链（CoT）来进行多步结构化推理，并在科学问答、符号数学、逻辑推演及程序合成等多种知识密集型应用场景中取得了显著进展。

尽管强化学习（RL）微调已取得显著进展，但其成效在很大程度上取决于训练数据的质量，这使得数据筛选工作日益受到重视。一种常见的做法是进行离线数据过滤，即在训练前利用预估难度、领域平衡度或多样性等静态启发式指标，对提示词（prompts）进行排序或筛选。然而，这种方法无法适应模型在训练过程中能力不断演变的情况。为了增强适应性，**近期研究探索了在线提示词选择策略，能够根据模型行为的变化动态调整选择**。这些方法通常以单步（per-step）或单轮（per-epoch）为单位进行操作，旨在挑选出能提供更强训练信号的信息量丰富的提示词。动态采样（Dynamic Sampling, DS）是一种具有代表性的前沿（SoTA）方法：它通过扩大候选提示词批次并为每个提示词生成多个回复，剔除那些奖赏稳定但信息量不足的提示词，最终利用保留下来的子集进行微调。该策略提升了训练样本的质量，并显著缩短了 RL 微调所需的训练步数。然而，对于推理密集型任务而言，生成包含长思维链（CoT）的回复计算成本高昂。因此，DS 方法因在扩大的批次上进行大规模大语言模型（LLM）生成而产生了巨大的额外开销，在实际应用中，这种开销往往超过了微调本身的成本。

**本研究旨在保持在线提示（prompt）选择的自适应性，同时避免冗余的试运行（rollout）**。为此，我们提出了 **Dynamics-Predictive Sampling (DPS)** 方法，通过推断提示的学习动态来在线预测具有信息量的提示。具体而言，我们引入了一种新视角，将强化学习（RL）微调过程中每个提示的求解进展形式化为一个动态系统。我们将每个提示的求解程度视为系统状态，并随着 LRM 的更新来追踪这些状态分布的演变。在技术实现上，我们将该过程建模为隐马尔可夫模型（HMM），以此作为追踪提示求解动态的可行工具。基于构建的动态系统，我们利用历史试运行的奖赏信号进行在线贝叶斯推断，从而估计演变中的状态分布。推断结果为自适应提示选择提供了预测性先验，进而提高了样本效率，无需依赖高强度的 rollout 筛选过程。

我们针对竞赛级数学、数值规划和视觉几何等多种下游推理任务，对所提出的 DPS 进行了实证评估。结果表明，DPS 能够准确预测提示词（prompts）不断演变的求解状态，并始终比基线方法筛选出更高比例的信息量丰富的样本。利用这一能力，DPS 显著加速了强化学习（RL）微调过程，仅需极少量的采样（rollouts）即可达到与“预知最优路径”（oracle）的高采样密集型策略 DS 相当甚至更优的性能。

## 2.PRELIMINARY

**RL Finetuning for LRMs**。给定从数据集 $\mathcal D$ 中采样得到的提示 $τ$ 以及由模型策略 $π_θ(y|τ)$ 生成的回复 $y$，强化学习（RL）微调的目标是最大化期望回报：

```math
\mathop{max}\limits_{\theta\in\Theta}\mathbb E_{τ\sim\mathcal D, y\sim\pi_{\theta}(\cdot|τ)}[r(τ,y)],\tag{1}
```

其中，奖励函数 $r(τ, y)$ 通常用于验证回答的正确性；在数学等领域，常采用二值信号（即正确为 1，错误为 0）。

**Group Relative Policy Optimization (GRPO)**。为了解决上述优化问题，人们提出了许多策略梯度方法。GRPO 是近年来广泛采用的一种变体，它无需显式估计价值函数，因此特别适用于 LLM 的微调。形式上，对于任意提示 $τ$ 及其对应的 $k$ 个采样响应 $\{y^τ_i\}^k_{i=1}$，GRPO 最大化以下目标函数：

```math
\begin{array}{cc}
\mathcal J_{GRPO}(\theta)=\mathbb E_{τ\sim\mathcal D,\{y^τ_i\}^k_{i=1}\sim\pi_{\theta_{old}}(\cdot|τ)}\\
\left[\frac{1}{k}\sum^k_{i=1}\left(min\left(\frac{\pi_{\theta}(y^τ_i|τ)}{\pi_{\theta_{old}}(y^τ_i|τ)}\hat A^τ_i,clip\left(\frac{\pi_{\theta}(y^τ_i|τ)}{\pi_{\theta_{old}}(y^τ_i|τ)}, 1-ϵ, 1+ϵ\right)\hat A^τ_i\right)-\beta D_{KL}(\pi_{\theta}||\pi_{ref})\right)\right],
\end{array}
```

其中，截断策略比率防止 $π_θ$ 与先前的策略 $π_{θ_{old}}$ 过度偏离，而正则化系数 $β$ 则惩罚与固定参考模型 $π_{ref}$ 的偏差。GRPO 采用基于组的归一化方案来估计优势 $\hat A^τ_i$：

```math
\hat A^τ_i=\frac{r(τ,y^τ_i)-mean(\{r(τ,y^τ_j)\}^k_{j=1})}{std(\{r(τ,y^τ_j)\}^k_{j=1})}.\tag{2}
```

该策略显著降低了训练复杂度，并在多种大语言模型（LLM）推理任务中展现出了强劲的实证表现。

**Dynamic Sampling for Online Prompt Selection**。在大语言模型（LLM）的强化学习微调过程中，不同训练样本对策略改进的贡献并不均等。当模型对某个问题始终给出正确或错误的回答时（这是训练期间常见的现象），奖赏信号所能提供的优化信息便十分有限。对于 GRPO 等算法而言，这种情况会导致归一化优势值（normalized advantages）趋于消失，从而实际上导致优化过程停滞。

为了缓解这一问题，人们提出了在线提示选择策略，旨在根据特定规则动态筛选提示。一种具有代表性的当前最优（SoTA）方法是动态采样（**Dynamic Sampling**，简称 DS）。在每个训练步骤 $t$，DS 首先基于随机采样构建一个较大的候选提示批次 $\hat{\mathcal B}_t$，随后剔除那些在 $k$ 个响应上奖赏值相同且不具信息量的提示，从而形成最终的训练批次 $\mathcal B_t$：

```math
\mathcal B_t=\left\{τ\in\hat{\mathcal B}_t|std(\{r(τ,y^τ_i)\}^k_{i=1})>0\right\}.\tag{3}
```

尽管 DS 行之有效，但由于需要对扩大的候选批次进行反复的 LLM 推理生成与评估，它也带来了显著的计算开销。在许多情况下，候选批次的规模数倍于最终批次，导致 LLM 生成成本成比例增加。对于需要生成长思维链（CoT）的推理任务而言，这种负担尤为突出。

关于相关工作的详尽讨论，请参阅附录 A。

## 3.DYNAMICS-PREDICTIVE SAMPLING FOR ACTIVE RL FINETUNING

<img
  src="https://i-blog.csdnimg.cn/direct/d1e147714bd64cc99a383ba04fa83a5c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

本节将提示（prompt）求解过程形式化为动态系统，针对范围预测问题制定了推理策略，并提出了一种用于在线提示选择的高效流水线。

### 3.1 GENERATIVE MODELING OF PROMPT-SOLVING DYNAMICS

**Problem Formulation**。既有研究表明，为了实现高效的策略优化，提示（prompt）存在着不同的求解状态。具体而言，历史重采样（History Resampling, HR）将提示分为“已完全求解”和“其他”两类，而 DS 方法则将“部分求解”的提示与其他提示区分开来。理论分析与实证结果均表明，那些既能产生成功响应又能产生失败响应的提示更具信息量，因为它们能为参数更新提供更强的梯度信号。基于此，本文为数据集 $\mathcal D$ 中的每个提示 $τ$ 定义了一个隐式状态 $z^τ_t ∈ \{1, 2, 3\}$，用于表征其在训练步骤 $t$ 的展开（rollout）结果：

- State 1 (fully unsolved)：所有回答均不正确，$\sum^k_{i=1} r(τ, y_i) = 0$;
- State 2 (partially solved)：有些回答正确，有些回答错误，$0 <\sum^k_{i=1} r(τ, y_i) < k$;
- State 3 (fully solved)：所有回答均正确，$\sum^k_{i=1} r(τ, y_i) = k$。

既有研究表明，State 2 的提示（prompt）包含最丰富的信息，因此在训练过程中应优先处理。然而，在每个训练步骤中，特定提示的求解状态在进行 rollout 与评估之前是未知的。在批量训练场景下，求解状态仅在特定提示被选中进行 rollout 时才会间歇性地被观测到；因此，每个提示都会产生一个间歇性的观测序列，其中提示 $\tau$ 在步骤 $t$ 的观测记为 $y^{\tau}_t$（若未进行观测，则 $y^{\tau}_t = \emptyset$）。**我们的目标是估计步骤 $t$ 在观测之前的求解状态**的滤波先验信念（filtered prior belief），记为 $\mu^{\tau, \text{prior}}_t$：

```math
\mu^{\tau, \text{prior}}_t(i):=\mathbb P(z^τ_t=i|y^τ_{1:t-1}),\quad\forall i\in\{1,2,3\},\tag{4}
```

**Prompt Solving as Dynamical Systems**。我们利用隐马尔可夫模型（HMM）对每个提示（prompt）求解状态的演变进行形式化建模，该模型刻画了在训练过程中，大语言模型（LLM）针对特定提示的求解能力如何发生变化。为表述清晰，我们在本节及第 3.2 节中省略了上标 $τ$，仅描述单个提示的生成与推理过程，该过程同样适用于所有其他提示。

形式上，初始求解状态 $z_1$ 采样自类别先验分布 $µ^{prior}_1 ∈ ∆^3$。在缺乏先验知识的情况下，我们采用均匀分布：

```math
z_1\sim Categorical(µ^{prior}_1),\quad µ^{prior}_1=[\frac{1}{3},\frac{1}{3},\frac{1}{3}].\tag{5}
```

后续状态根据马尔可夫过程演化，其转移矩阵 $Φ ∈ \mathbb R^{3\times 3}$ 为列随机矩阵，其中元素 $Φ(i, j)$ 表示从状态 $j$ 转移到状态 $i$ 的概率：

```math
z_t|z_{t-1}\sim Categorical(Φ(\cdot,z_{t-1})),\quad Φ(i,j)=\mathbb P(z_t=i|z_{t-1}=j),\quad\sum^3_{i=1}Φ(i,j)=1.\tag{6}
```

在每个时刻 $t$，若该提示（prompt）被选中用于训练，观测值 $y_t$ 便能精确揭示当前状态；否则，状态保持未被观测。这导致了一个退化的发射模型：

```math
p(y_t|z_t)=\begin{cases}
\delta(y_t,z_t), & if~y_t\in\{1,2,3\},\\
1, & if~y_t=\emptyset,
\end{cases}\tag{7}
```

其中，$\delta(\cdot,\cdot)$ 表示 Kronecker delta 函数。将缺失观测的发射概率设为 1，可以在不对 $z_t$ 施加任何约束的同时保持边缘一致性。将这些组成部分结合起来，每个提示的求解过程可以表示为一个动态系统。具体而言，状态 $z_{1:T}$ 和观测 $y_{1:T}$ 上的联合分布可以分解为：

```math
p(z_{1:T},y_{1:T})=
\int p(z_1)
\prod_{t=2}^{T} p(z_t\mid z_{t-1},\Phi)
\prod_{t=1}^{T} p(y_t\mid z_t)\,\mathrm{d}\Phi,
\tag{8}
```

其中，转移矩阵 $\Phi$ 被视为一个随机变量。该形式化方法刻画了底层的生成过程，从而能够进一步对求解状态进行贝叶斯推断。

### 3.2 ONLINE INFERENCE AND TRANSITION LEARNING

我们在训练过程中执行在线贝叶斯推断，以跟踪给定提示（prompt）的求解状态。该过程在每个训练步骤 \(t\) 遵循一个三阶段流程：（i）如果观测 $y_t$ 可用，则利用 $y_t$ 将先验 $\mu_t^{\mathrm{prior}}$ 更新为后验 $\mu_t^{\mathrm{post}}$；否则，将后验设为先验；（ii）如果观测到了 $y_t$，则进一步更新转移模型；以及（iii）通过转移模型将后验向前传播，以生成下一步的先验 $\mu_{t+1}^{\mathrm{prior}}$。

**Observation Update**。如果观测到了 $y_t$，则根据贝叶斯法则，将先验 $\mu_t^{\mathrm{prior}}$ 更新为后验 $\mu_t^{\mathrm{post}}$：

```math
\mu_t^{\mathrm{post}}(i)=
\frac{p(y_t\mid z_t=i)\,\mu_t^{\mathrm{prior}}(i)}
{\sum_k p(y_t\mid z_t=k)\,\mu_t^{\mathrm{prior}}(k)}=
\frac{\delta(y_t,i)\cdot\mu_t^{\mathrm{prior}}(i)}
{\sum_k \delta(y_t,k)\cdot\mu_t^{\mathrm{prior}}(k)},
\qquad
\text{若 } y_t\in\{1,2,3\}.
\tag{9}
```

如果 $y_t$ 未被观测到，则在没有新证据的情况下，贝叶斯更新退化为

```math
\mu_t^{\mathrm{post}}=\mu_t^{\mathrm{prior}}.
```

**Transition Update**。我们对转移矩阵的各列设置相互独立的 Dirichlet 先验：

```math
\Phi_t(\cdot,j)
\sim
\operatorname{Dirichlet}
\bigl(
\alpha_t(1,j),
\alpha_t(2,j),
\alpha_t(3,j)
\bigr),
\qquad
\forall j\in\{1,2,3\},
\tag{10}
```

其中，$\alpha_t(i,j)$ 用于刻画转移概率上的分布。我们通过设置 $\alpha_0(i,j)=1$，使用一个无信息先验（uninformative prior）来初始化转移矩阵。随着观测数据依次到达，参数 $\alpha_t(i,j)$ 将进行在线更新。具体而言，当在步骤 $t$ 观测到 $y_t$ 时，利用软转移统计量（soft transition statistics）对 $\alpha_t(i,j)$ 进行贝叶斯更新：

```math
\alpha_t(i,j)=\alpha_{t-1}(i,j)+\xi_t(i,j),
\tag{11}
```

其中，$\xi_t(i,j)$ 表示后验转移伪计数（posterior transition pseudo-count）：

```math
\xi_t(i,j)
:=
\mathbb{P}(z_{t-1}=j,z_t=i\mid y_{1:t}),
\qquad
\text{若 } y_t\in\{1,2,3\}.
\tag{12}
```

这一更新规则来源于 Dirichlet 分布与 Categorical 分布之间的共轭性。观测到一次从状态 $j$ 到状态 $i$ 的转移，会为 Dirichlet 先验中相应的参数增加一个伪计数。由于实际的状态转移具有不确定性，其期望贡献由 $\xi_t(i,j)$ 给出。根据马尔可夫性质以及“给定状态后各观测条件独立”这一性质，我们得到：

```math
\mathbb{P}(z_{t-1}=j,z_t=i\mid y_{1:t})=
\frac{
\mu_{t-1}^{\mathrm{post}}(j)
\cdot
\Phi_{t-1}(i,j)
\cdot
p(y_t\mid z_t=i)
}{
\sum_{j'}
\mu_{t-1}^{\mathrm{post}}(j')
\sum_{i'}
\Phi_{t-1}(i',j')
\cdot
p(y_t\mid z_t=i')
}.
\tag{13}
```

具体推导见第 C 节。使用式（7）中的确定性发射模型（deterministic emission model），并在 \(y_t\) 未被观测时令 $\xi_t=0$（因此贝叶斯更新退化为先验），则 $\xi_t$ 可以简化为：

```math
\xi_t(i,j)=
\begin{cases}
\dfrac{
\mu_{t-1}^{\mathrm{post}}(j)\cdot\Phi_{t-1}(i,j)
}{
\sum_{j'}
\mu_{t-1}^{\mathrm{post}}(j')
\cdot
\Phi_{t-1}(i,j')
},
&
\text{若 } i=y_t,
\\[12pt]
0,
&
\text{其他情况}.
\end{cases}
\tag{14}
```

*Non-stationary Extension*。标准的贝叶斯 HMM 假设状态转移动态是平稳的（stationary）。然而，由于复杂的学习过程，大语言模型（LRMs）中的提示求解状态可能会以非平稳的方式演化。为了适应不断变化的转移动态，我们提出了一种轻量级扩展：对转移模型采用具有指数衰减的 Dirichlet 后验更新：

```math
\alpha_t(i,j)=
\lambda\cdot\alpha_{t-1}(i,j)
+
(1-\lambda)\cdot\alpha_0(i,j)
+
\xi_t(i,j),
\qquad
\lambda\in(0,1).
\tag{15}
```

该机制通过更强调近期的转移统计量，同时逐渐降低较旧模式的权重，从而引入“遗忘”机制。较小的 $\lambda$ 值能够使模型更快地适应不断变化的动态。先验 $\alpha_0$ 起到正则化器的作用：当近期证据较为稀疏时，它可以防止模型发生退化，同时还允许将关于合理转移结构的领域知识编码到模型中。

**Next-state Prediction**。 在步骤 $t$ 完成观测更新和转移更新之后，我们使用后验信念 $\mu_t^{\mathrm{post}}$ 以及推断得到的转移矩阵 $\Phi_t$，构造下一步的预测先验：

```math
\mu_{t+1}^{\mathrm{prior}}=
\Phi_t \mu_t^{\mathrm{post}},
\qquad
\text{即，}\quad
\mu_{t+1}^{\mathrm{prior}}(i)=
\sum_{j=1}^{3}
\Phi_t(i,j)\cdot \mu_t^{\mathrm{post}}(j).
\tag{16}
```

该先验 $\mu_{t+1}^{\mathrm{prior}}$ 表示在训练步骤 $t+1$ 的观测到来之前，我们对提示求解状态（prompt-solving state）的预测，并作为后续推断迭代中的初始信念。

不同于经典的 HMM 平滑方法（例如 Forward-Backward，Baum et al., 1972），后者需要访问完整的状态轨迹，我们的方法以在线方式同时更新状态信念和转移分布的后验。

此外，与响应生成（response rollout）或模型微调相比，该推断框架的计算开销通常可以忽略不计，因为它仅涉及非常低维的矩阵运算。

### 3.3  PROMPT SAMPLING WITH PREDICTED DYNAMICS

<img
  src="https://i-blog.csdnimg.cn/direct/5dff45c8c73c46efa6b2214e8d69a7aa.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

对提示求解动态进行建模的核心目标，是在每个训练步骤中、在执行代价高昂的 rollout 之前，**在线预测哪些提示（prompts）应被优先用于训练**。对于每个提示 $\tau$，给定其预测求解状态信念

```math
\mu_t^{\tau,\mathrm{prior}}=
\mathbb{P}(z_t\mid y_{1:t-1}),
```

我们根据该提示被预测为**部分解决（State 2）**的概率来确定其优先级，该概率记为

```math
\mu_t^{\tau,\mathrm{prior}}(2).
```

关键的是，我们使用的是先验信念 $\mu_t^{\tau,\mathrm{prior}}$，而不是后验信念 $\mu_t^{\tau,\mathrm{post}}$，因为提示的选择必须发生在通过 rollout 观测到步骤 $t$ 的结果之前。形式化地，在步骤 $t$，选择 State 2 概率最高的 $B$ 个提示，以构成当前训练批次：

```math
\mathcal{B}_t=
\operatorname{Top}_B
\left(
\left\{
\tau\in\mathcal{D}
\mid
\mu_t^{\tau,\mathrm{prior}}(2)
\right\}
\right).
\tag{17}
```

我们注意到，尽管 Top-$B$ 选择策略本身是一种纯粹的**利用（exploitation）**策略，但它所利用的目标量（即 $\mu_t^{\tau,\mathrm{prior}}$）已经通过非平稳衰减机制包含了一定程度的**探索（exploration）**。

**Overall Algorithm**。将上述各个组成部分整合起来，我们在算法 1 中给出了完整的 DPS 算法，其整体框架概览如图 1 所示。关于 DPS 时间复杂度的详细分析，以及它与课程学习（curriculum learning）之间隐含联系的讨论，见附录 B。

## 4.EXPERIMENTS

在本节中，我们进行了多项实验以验证 DPS 的有效性。附录 D、E 和 F 分别提供了实现细节、补充结果及数据示例。
