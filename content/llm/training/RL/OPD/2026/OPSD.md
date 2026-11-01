# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models翻译

论文链接：https://arxiv.org/pdf/2601.18734

代码链接：https://github.com/siyan-zhao/OPSD

## 摘要

知识蒸馏通过将“教师”大语言模型（LLM）的知识压缩并用于训练较小的“学生”LLM，从而提升后者的推理能力。在线（on-policy）蒸馏进一步优化了这一流程：学生模型在采样自身推理轨迹的同时，由教师模型提供细粒度的（token 级别的）监督信号，从而解决了离线（off-policy）蒸馏方法中训练阶段与推理阶段分布不一致的问题。然而，在线蒸馏通常需要一个额外的（往往规模更大的）教师模型，且未能充分利用推理数据集中的标准答案（ground-truth solutions）。受此启发——**即一个能力足够强的大模型能够理解外部的“特权”推理轨迹，并以此指导能力较弱的自身**——我们提出了“在线自蒸馏”（On-Policy Self-Distillation, OPSD）算法。在该算法中，同一个大模型在不同的上下文条件下，既充当教师，也充当学生。教师策略基于“特权信息”（如经过验证的推理轨迹）进行决策，而学生策略仅根据问题本身进行决策；训练过程旨在最小化学生模型在自身生成的推理轨迹上，这两种策略分布之间的 token 级差异。我们在多个数学推理基准测试中验证了该方法的有效性：与强化学习方法相比，它具有更高的 token 使用效率；与离线蒸馏方法相比，它表现出更优的性能。代码仓库：https://github.com/siyan-zhao/OPSD。

## 1.Introduction

<img
  src="https://i-blog.csdnimg.cn/direct/d6bf3f4228a94af6b5e63789f72d6541.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

大语言模型（LLM）近期的进展展现出了令人瞩目的推理与指令遵循能力。在后训练阶段实现这些能力，通常依赖于强化学习方法（如基于可验证奖励的强化学习，即 RLVR，典型代表为 GRPO）、在高质量推理数据集上进行的监督微调（SFT），或是知识蒸馏。近期研究表明，通过从先进的“教师模型”进行蒸馏，所取得的效果在模型性能和训练效率上均可优于强化学习方法。

尽管各类方法都取得了一定的成功，但它们各自也存在固有的局限性。RLVR（基于强化学习的验证与奖励）存在效率低下的问题，具体表现为：(1) 针对每个提示（prompt）采样多个响应在计算上开销巨大，且在估计真实价值函数时可能引入较大的方差；此外，当所有采样结果全为正确或全为错误时，梯度信号会消失。(2) 奖励信号较为稀疏，且被统一应用于生成输出中的所有 token，忽略了细粒度​​的 token 级反馈。有监督微调（SFT）则面临“暴露偏差”（exposure bias）问题，且泛化能力较弱。传统的知识蒸馏虽然能通过教师模型提供密集的 token 级监督，但依赖于“离线策略”（off-policy）数据。近期，“在线策略”（on-policy）蒸馏技术取得了进展——即学生模型采样自身的轨迹，同时由教师策略提供密集的 token 级监督——该方法结合了同策略训练的分布真实性与​​密集的反馈信号，展现出了更优的样本效率。

尽管“在线策略蒸馏”（on-policy distillation）已展现出优异的性能，但它通常依赖一个独立的“教师模型”来指导“学生模型”。鉴于现代大语言模型（LLM）已具备强大的推理能力，我们提出了这样一个研究问题：**模型能否通过“自蒸馏”（self-distillation）机制，让自己担任自己的教师**？我们的方法受到人类学习过程的启发：当学生解题出错时，他们可以通过查看正确答案、梳理其推理步骤，从而找出自己推理过程中的症结所在。既往研究表明，对于大语言模型而言，评估任务往往比生成任务更容易。我们据此推测，“合理化解释”（即针对给定的正确答案进行逻辑阐述）同样比直接生成答案更为简单。基于这一思路，我们让教师策略和学生策略均源自同一个大语言模型。教师策略在执行时可获取“特权信息” $y^\star$（例如标准答案或参考思维链），而学生策略则仅根据问题 $x$ 进行推理。具体而言，教师策略 $p_T(\cdot | x, y^\star)$ 同时依赖于问题和特权答案，而学生策略 $p_S(\cdot | x)$ 仅能观测到问题本身。我们保留了“在线策略”训练范式：通过学生策略采样生成轨迹 $\hat{y}$，并由教师策略提供细粒度的、基于 token 级别的监督信号。

因此，我们提出了 **On-Policy Self-Distillation (OPSD)** 框架。在该框架中，单一模型同时扮演教师和学生两种角色。学生模型采样生成自身的轨迹 $\hat{y} \sim p_S(\cdot | x)$；随后，我们计算学生模型与教师模型分布在每个 token 上的散度，并在学生模型自身的采样轨迹上最小化该散度。这种方法具有以下特点：(i) 采用 on-policy 监督（即使用学生模型自身的轨迹）；(ii) 提供针对每个 token 的密集反馈；(iii) 利用了真实标准答案 $y^\star$；(iv) 无需额外的教师模型。整个学习过程通过损失函数来体现：

```math
\mathcal L_{OPSD}=\mathbb E_{(x,y^*)\sim\mathcal S}\mathbb E_{\hat y\sim p_{S}(\cdot|x)}\sum^{|\hat y|}_{n=1}D\left(p_T(\cdot|x,y^*,\hat y_{\lt n})||p_S(\cdot|x,\hat y_{\lt n})\right).\tag{1}
```

总之，我们的贡献如下：
- 我们提出了 On-Policy Self-Distillation (OPSD)，这一新框架。该框架使单一模型能够同时扮演教师和学生角色，并利用真实答案（ground-truth answers）为学生的输出序列提供细粒度的（token 级别）监督信号。
- 我们引入了一种针对每个 token 的逐点 KL 截断机制；该机制能够稳定训练过程并提升模型性能，因为我们发现，风格类 token 可能会主导数学类 token 的训练信号。
- 我们在三项竞赛级数学推理任务上对 OPSD 进行了评估，结果表明，OPSD 在达到与 GRPO 相当的性能水平的同时，显著提升了 Token 效率，且优于监督微调（SFT）。
- 我们分析了不同散度目标的影响、学生模型生成长度的作用，以及学生模型与教师模型在生成风格上的差异。

## 2.Background

### 2.1 Knowledge Distillation for Autoregressive Large Language Models

知识蒸馏通过训练较小的“学生”模型去模仿较大的“教师”模型的行为，从而将知识从教师模型迁移到学生模型中。其核心思想在于，教师模型在各类别上的“软”概率分布（soft probability distribution）所包含的信息，要比单纯的“硬”标签（hard labels）更为丰富；这是因为软概率分布揭示了教师模型所学到的各类别之间的相似性。对于自回归语言模型，假设给定一个数据集 $\mathcal S = \{(x, y^\star)\}$，其中 $x$ 表示输入，$y^\star$ 为对应的参考输出；此时，教师模型 $p_T$ 和学生模型 $p_S$ 均在词表 $\mathcal V$ 上定义了针对每个 token 的概率分布。传统的监督式知识蒸馏旨在最小化教师模型分布与学生模型分布之间的某种散度（divergence）$D$，并对固定数据集上的该散度值取平均：

```math
\mathcal L_{\text{Supervised Distillation}}(\theta)=\mathbb E_{(x,y)\sim\mathcal S}[D(p_T||p_S)(y|x)],\tag{2}
```

其中 $D(p_T \parallel p_S)(y|x) = \frac{1}{|y|} \sum_{n=1}^{|y|} D(p_T(\cdot|y_{<n}, x) \parallel p_S(\cdot|y_{<n}, x))$ 用于衡量逐个 token 的差异。然而，这种“离线策略”（off-policy）方法存在分布不匹配的问题：在推理阶段进行自回归生成时，学生模型所遇到的部分序列 $y_{<n}$，与训练阶段在固定数据集上所见到的序列不同，这会导致误差累积。而“在线策略”（on-policy）蒸馏通过让学生模型基于其自身生成的序列 $\hat{y} \sim p_S(\cdot|x)$ 进行训练来解决这一问题；在此过程中，学生模型可以针对这些“在线策略”样本，从教师模型处获得密集的 token 级反馈：

```math
\mathcal L_{\text{On-policy Distillation}}(\theta)=\mathbb E_{x\sim\mathcal S}[\mathbb E_{\hat y\sim p_S(\cdot|x)}[D(p_T||p_S)(y|x)]],\tag{2}
```

该方法将知识蒸馏与模仿学习联系起来：学生模型通过学习教师模型针对其自身输出所给出的指导，不断进行迭代改进。这种方式既结合了强化学习中“在线策略”（on-policy）的相关性，又融合了监督学习中稠密的奖赏信号，从而在保持计算效率的同时，缓解了“暴露偏差”（exposure bias）问题。

### 2.2 Reinforcement Learning with Verifiable Rewards

“带可验证奖赏的强化学习”（RLVR）已成为大语言模型后训练阶段的一种主流方法。该方法常用于数学和编程等结果易于验证的任务中，并采用近端策略优化（PPO）和组相对策略优化（GRPO）等算法来实现。

在 GRPO 训练过程中，对于每个问题 $x$，从当前策略 $\pi_\theta$ 中采样出一组 $G$ 个回答 $\{o_1, o_2, \dots, o_G\}$。每个回答 $o_i$ 都会获得一个二元奖励 $r_i \in \{0, 1\}$，用于指示其正确性。随后，该方法利用“组归一化奖赏”（group-normalized reward），为回答 $o_i$ 中的所有 token $k = 1, \dots, |o_i|$ 分配优势值（advantage）：

```math
A_i=\frac{r_i-mean(\{r_j\}^G_{j=1})}{std(\{r_j\}^G_{j=1})}.\tag{4}
```

这一公式可以从价值函数的视角来理解：均值 $\text{mean}(\{r_j\}_{j=1}^G)$ 可视为价值函数 $V(x)$ 的 $G$ 样本蒙特卡洛估计，而稀疏的二元奖励 $r_j$ 则代表了（未折扣的）状态-动作价值 $Q(x, o_j)$。关键在于，由于奖赏信号仅在整个序列层面给出，因此响应中的所有 token 共享相同的优势值（advantage）。GRPO 的目标函数引入了截断式代理损失（clipped surrogate loss）来调节策略更新，并辅以反向 KL 散度惩罚，以防止策略过度偏离参考策略。

```math
\mathcal L_{GRPO}(\theta)=\mathbb E_{x\sim\mathcal S,o_1,...,o_G\sim\pi_{\theta}(\cdot|x)}\left[\frac{1}{G}\sum^G_{i=1}\frac{1}{|o_i|}\sum^{|o_i|}_{n=1}min(\rho^n_iA_i,clip(\rho^n_i,1-ε,1+ε)A_i)-\beta D_{KL}[\pi_{\theta}(\cdot|x)||\pi_{ref}(\cdot|x)]\right]\tag{5}
```

其中 $ρ^n_i = \frac{π_θ(o^n_i|x,o^{<n}_i)}{π_{θ_{old}}(o^n_i|x,o^{<n}_i)}$ 为重要性比率，$π_{θ_{old}}$ 表示更新前的策略，而 $ε$ 则用于控制截断范围。

尽管 RLVR 方法在实践中表现出色，但它们仍面临两个主要局限：(1) 奖赏信号稀疏，仅能提供序列层面的反馈，无法在 Token 层面指出错误发生的位置；(2) 当所有采样得到的回复获得相同的奖赏（即全对或全错）时，其优势便荡然无存，导致尽管付出了采样计算成本，却无法进行任何策略更新。

## 3.Methods

### 3.1 Learning from Verifiable Reasoning Dataset

<img
  src="https://i-blog.csdnimg.cn/direct/48abf880ace04567975dc737d154335c.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

我们考虑一个由“问题-解答”对组成的数据集 $\mathcal S = \{(x_i, y^\star_i)\}_{i=1}^N$，其中 $x_i$ 表示一个问题，$y^\star_i$ 是对应的参考解答（可能包含思维链推理过程）。为简洁起见，我们省略样本索引 $i$，用 $(x, y^\star)$ 来表示数据集中的一个通用样本。我们可以通过多种方式利用该数据集中的学习信号：在 $\mathcal S$ 上进行标准的有监督微调（SFT）可被视为一种基于专家轨迹的“离线策略”（off-policy）蒸馏或模仿学习，但它存在训练与推理阶段分布不一致的问题。基于可验证奖赏的强化学习（RLVR，例如 GRPO）通过优化“在线策略”（on-policy）样本来解决这一问题；它通过将模型生成的答案与参考解答 $y^\star$ 进行比对，从而赋予二元奖赏。然而，RLVR 的计算成本高昂，且奖赏信号较为稀疏——无论错误发生在何处，它对所有生成的 token 都给予相同的反馈。另一种方案是训练一个过程奖赏模型（PRM），以便在强化学习过程中提供密集的、细化到 token 级别的反馈。但获取用于训练 PRM 的标签代价极高，且难以扩展。在线策略蒸馏方法通过让模型在自身生成的样本上进行训练来缓解分布偏移问题，但这类方法通常需要一个额外的（且往往规模更大的）教师模型来提供监督信号。我们则寻求一种既密集又属于“在线策略”范畴的训练信号，且该信号无需依赖外部教师模型或奖赏模型。这促成了我们提出的“在线策略自蒸馏”（On-Policy Self-Distillation）方法。我们在表 1 中总结了这些方法之间的区别。

### 3.2 On-Policy Self-Distillation

<img
  src="https://i-blog.csdnimg.cn/direct/f909e5d2278d448482a5f6f20a5b0421.png"
  alt=""
  referrerpolicy="no-referrer"
  style="max-width: 100%; height: auto;"
/>

**Motivation: Learning by understanding solutions**。我们提出了一种受学生学习方式启发的全新视角：当学生在解决问题遇到困难时，他们不必进行漫长的反复试错，而是可以直接查看解法、理解其中的逻辑推理，并将这种解题思路内化。同样地，如果模型能够获取正确答案或推理过程 $y^\star$，且自身具备足够的能力，它便能理清推理步骤并实现“自我教学”——这类似于学生通过回顾解法来推导其为何有效。这一直觉构成了我们框架的基础：我们在训练过程中直接利用标准答案（即“真值”解法）$y^\star$ 作为特权信息，使模型能够充当自己的“教师”，而无需依赖外部奖赏模型或规模更大的教师模型。

**Teacher and student policies**。我们通过改变条件上下文，从同一个语言模型 $p_θ$ 实例化两个条件分布。教师策略基于特权信息——问题 $x$ 和参考解 $y^{\star}$：

```math
p_T(\cdot|x,y^{\star})≜p_{\theta}(\cdot|x,y^{\star}).
```

学生策略仅关注问题陈述，这与推理阶段的条件相契合：

```math
p_S(\cdot|x)≜p_{\theta}(\cdot|x).
```

这两种策略共享相同的参数 $θ$，仅在各自的条件上下文（conditioning context）上有所不同。为了引导“教师”模型自然地评估“学生”模型所生成的内容，我们加入了一个提示（prompt），要求“教师”模型在查看参考解法后生成一个新的解法，如图 2 所示。不过，“教师”模型并不会实际生成 token，而是通过“预填充”（prefilling）过程隐式地进行推理阐述（rationalization）。

**On-policy sampling from the student**。给定问题 $x$，学生生成一个在线策略（on-policy）响应。

```math
\hat y=(\hat y_1,...,\hat y_{|\hat y|})\sim p_S(\cdot|x).
```

随后，这两种策略会对该由学生生成的轨迹进行评估。在每个位置 $n$，它们基于相同的前缀（由学生生成），推导出关于下一个 token 在词表 $\mathcal V$ 上的分布（即 $y_n ∈ \mathcal V$ 的分布）：

```math
p_S(y_n|x,\hat y_{<n}),\quad p_T(y_n|x,y^{\star},\hat y_{\lt n}),
```

其中，$\hat y_{\lt n}≜(\hat y_1,...,\hat y_{n-1})$。

**Training objective: Full-vocabulary logit distillation**。我们实例化了一个全词表散度目标，该目标在每个位置上匹配教师和学生的下一个 token 分布。给定一个学生生成的序列 $\hat y$，定义轨迹平均的逐 token 差异。

```math
D(p_T||p_S)(\hat y|x)≜\frac{1}{|\hat y|}\sum^{|\hat y|}_{n=1}D\left(p_T(\cdot|x,y^{\star},\hat y_{\lt n})||p_S(\cdot|x,\hat y_{\lt n})\right),\tag{6}
```

其中 $p_S(\cdot \mid x, \hat{y}_{<n})$ 和 $p_T(\cdot \mid x, y^\star, \hat{y}_{<n})$ 表示在下一个 token $y_n \in V$ 上的分布。此处，$D$ 可以是任意分布散度度量，例如广义 Jensen-Shannon 散度 $JSD_\beta$；对于权重 $\beta \in [0, 1]$，其定义如下：

```math
JSD_{\beta}(p_T||p_S)=\beta D_{KL}(p_T||m)+(1-\beta)D_{KL}(p_S||m)\tag{7}
```

其中 $m = βp_T + (1−β)p_S$ 为插值混合分布。这种全词表形式提供了密集的、token 级别的反馈：教师模型在获知 $y^{\star}$ 的前提下，向学生模型展示所有可能的后续 token 分布，并引导其走向能够得出正确答案的推理路径。

我们最小化教师模型与学生模型在“在线策略”（on-policy）学生样本上的预期散度：

```math
\mathcal L(\theta)=\mathbb E_{(x,y^{\star})\sim\mathcal S}[\mathbb E_{\hat y\sim p_S(\cdot|x)}[D(p_T||p_S)(\hat y|x)]].\tag{8}
```

梯度仅通过学生策略 $p_S$ 进行反向传播，而教师策略 $p_T$ 则作为基于特权信息 $(x, y^\star)$ 的固定全分布目标。

**Per-Token Pointwise Divergence Clipping**。在我们的实验中，我们观察到 token 层面的散度在不同词表项之间分布极不均衡：一小部分“风格型” token 所表现出的散度，远高于那些具有数学意义的 token（见表 5）。这种不均衡导致训练信号被风格模式所主导。为解决这一问题，我们对词汇层面的散度贡献值采用了逐点截断（pointwise clipping）处理。设 $D_f(p_T \| p_S)$ 为一种 $f$-散度。对于每个 token 位置 $n$ 及词表项 $v$，定义如下：

```math
ℓ^{(f)}_{n,v}=p_T(v|\cdot)f\left(\frac{p_S(v|\cdot)}{p_T(v|\cdot)}\right).
```

我们计算截断散度：

```math
D^{(f)}_{clip}(p_T||p_S)=\frac{1}{|\hat y|}\sum^{|\hat y|}_{n=1}\sum_{v\in\mathcal V}min(ℓ^{(f)}_{n,v},\tau).
```

**Alternative objective: Sampled-token distillation through policy gradient**。借鉴近期的z在线策略（on-policy）蒸馏方法，我们构建了一种“采样 token 奖赏信号”（即针对采样动作的逆向 KL 散度信号），并利用策略梯度进行优化。对于采样序列 $\hat{y}$ 中的每个位置 $n$，定义优势项（advantage term）如下：

```math
A_n(x,\hat y)=log~p_T(\hat y_n|x,y^{\star},\hat y_{\lt n})-log~p_S(\hat y_n|x,\hat y_{\lt n}),
```

并优化策略梯度形式的目标函数：

```math
\mathcal L(\theta)=\tag{9}
```

**OPSD as dense-reward policy gradient and comparison to STaR**。公式 (9) 中的目标函数可被视为一种具有密集型、词元级（token-level）奖励的策略梯度（policy gradient）方法。我们在附录 D 节中对此进行了形式化定义，并将其与 STaR 方法进行了对比。STaR 是一种密切相关的方法，它同样利用同一模型生成推理轨迹，随后进行拒绝采样（rejection sampling），并针对正确的推理轨迹执行有监督微调（SFT）。这一过程可被视为一种基于序列级二元奖励的策略梯度方法——它为所有词元赋予相同的贡献权重，而当样本结果错误时，该奖励则归零。相比之下，OPSD 能够在每一个 token 位置提供反馈，且无论最终答案是否正确。

## 4.Experiments
